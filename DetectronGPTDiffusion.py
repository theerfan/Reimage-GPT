import os
import cv2
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from diffusers import DiffusionPipeline

from torch import nn
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import pathlib
import matplotlib.pyplot as plt

# Import the necessary libraries for partitioning the dataset into train and test sets
from sklearn.model_selection import train_test_split


class DetectronGPTDiffusion(nn.Module):
    def __init__(self):
        super(DetectronGPTDiffusion, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detectron2 parts
        # Note that the "items" we could detect here are limited to the 80 classes
        # of the COCO dataset, but we can add more classes if we want to.
        self.cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running
        # a model in detectron2's core library
        self.cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        )
        # Detection threshold for this model
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        # Find a model from detectron2's model zoo.
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        # This is the main detectron module
        # This has no .parameters() function so we assume it's already frozen
        self.detectron = DefaultPredictor(self.cfg)

        # GPT-J parts
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.gpt = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", cache_dir=pathlib.Path("cache").resolve()
        ).to(self.device)

        # Diffusion parts
        # This has no .parameters() function so we assume it's already frozen
        self.diffusion_model = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(self.device)

    def partition_image(self, image_dir: str = "input.jpg"):
        im = cv2.imread(image_dir)
        # Round down the image's height and width to the nearest multiple of 8
        # (the largest multiple of 8 that is smaller than the image's height and width)
        im_height = im.shape[0]
        im_width = im.shape[1]
        im_height = im_height - (im_height % 8)
        im_width = im_width - (im_width % 8)
        im = cv2.resize(im, (im_width, im_height))

        outputs = self.detectron(im)
        predictions = outputs["instances"]

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = (
            predictions.pred_classes.tolist()
            if predictions.has("pred_classes")
            else None
        )
        labels = _create_text_labels(classes, self.metadata.get("thing_classes", None))
        keypoints = (
            predictions.pred_keypoints if predictions.has("pred_keypoints") else None
        )

        return im, [boxes, scores, classes, labels, keypoints]

    def gpt_post_process(self, output_sequences, input_prompt):
        predictions = []
        generated_sequences = []

        # decode prediction
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(
                generated_sequence,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            # Take out the prompt given to it by us
            text = text[len(input_prompt) :]
            generated_sequences.append(text.strip())

        for i, g in enumerate(generated_sequences):
            res = str(g).replace("\n\n\n", "\n").replace("\n\n", "\n")
            lines = res.split("\n")
            predictions.append("\n".join(lines))

        return predictions

    # temp = min:0, max:3, step:0.01
    # top_p = # min:0, max:1, step:0.01
    # repetition penalty is very critical for GPT-2
    # TODO: Move these gpt generation parameters to the __init__ function
    def gpt_generate(
        self,
        prompt,
        num_sequences=1,
        min_length=128,
        max_length=256,
        temperature=1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.0,
    ):

        prompt_full = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        encoded_prompt = prompt_full.input_ids
        encoded_prompt = encoded_prompt.to(self.device)

        output_sequences = self.gpt.generate(
            input_ids=encoded_prompt,
            max_new_tokens=max_length,
            attention_mask=prompt_full.attention_mask,
            min_length=min_length,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=True,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_sequences,
        )

        return self.gpt_post_process(output_sequences, prompt)

    def generate_image(self, prompt, height=256, width=256):
        max_length = self.diffusion_model.tokenizer.model_max_length
        input_ids = self.diffusion_model.tokenizer(
            prompt, return_tensors="pt"
        ).input_ids
        input_ids = input_ids.to(self.device)

        negative_ids = self.diffusion_model.tokenizer(
            "",
            truncation=False,
            padding="max_length",
            max_length=input_ids.shape[-1],
            return_tensors="pt",
        ).input_ids
        negative_ids = negative_ids.to(self.device)

        concat_embeds = []
        neg_embeds = []
        for i in range(0, input_ids.shape[-1], max_length):
            concat_embeds.append(
                self.diffusion_model.text_encoder(input_ids[:, i : i + max_length])[0]
            )
            neg_embeds.append(
                self.diffusion_model.text_encoder(negative_ids[:, i : i + max_length])[
                    0
                ]
            )

        prompt_embeds = torch.cat(concat_embeds, dim=1)
        negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

        return self.diffusion_model(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            height=height,
            width=width,
        ).images

    def generate_description(self, boxes, labels):
        full_str = ""
        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]
            box_array = box.tensor.flatten().tolist()
            # Just the pixel is enough, no floating points
            box_array = [int(x) for x in box_array]
            full_str += f"a {label} at {box_array}, "
        return full_str

    def forward(self, image_dir: str):
        input_image, [
            boxes,
            scores,
            classes,
            labels,
            keypoints,
        ] = self.partition_image(image_dir)
        height = input_image.shape[0]
        width = input_image.shape[1]
        description = self.generate_description(boxes, labels)
        prompt = (
            "We would like to generate a text prompt that will be used to generate an image. In this image, [a, b, c, d] represents the position of the object in the image. In the prompt, describe the objects to be included as well as their positions."  # noqa: E501
            + description
        )
        engineered_prompt = self.gpt_generate(prompt)[0]
        output_image = self.generate_image(engineered_prompt, height, width)[0]
        # print(prompt)
        # print(engineered_prompt)
        return {
            "manual_prompt": prompt,
            "output_prompt": engineered_prompt,
            "input_image": input_image,
            "output_image": output_image,
        }


def _create_text_labels(classes, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]

    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels


import torchvision.transforms as transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim

im_transform = transforms.ToTensor()


def ssim_loss(input_image, output_image):
    # Calculate the score between the input and output images
    input_torch = im_transform(input_image)
    output_torch = im_transform(output_image)
    score = ssim(input_torch.unsqueeze(0), output_torch.unsqueeze(0))
    return 1 - score


# A function which returns the filenames of all the images in a directory
def get_image_filenames(directory):
    # Get all the filenames
    filenames = os.listdir(directory)
    # Filter out the non-image files
    filenames = [f for f in filenames if f.endswith(".jpg")]
    # Add the directory to the filenames
    filenames = [os.path.join(directory, f) for f in filenames]
    return filenames


# Optimize the parameters of the gpt part of the model
def optimize_gpt(
    model: DetectronGPTDiffusion,
    n_epochs=1,
    # if None selects the whole dataset, otherwise it'll use the first `dataset_size` points of it.
    dataset_size=None,
    learning_rate=0.01,
):
    # Get the dataset
    dataset = get_image_filenames(
        "/content/drive/Shareddrives/COM SCI 263/Final Project/Data/COCO/val2017"
    )[:dataset_size]
    # We use run the model one time for each image in the dataset
    # (Filtered to only n_steps images)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    # Define the optimizer
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.gpt.parameters(),
                "lr": learning_rate,
            },
        ]
    )

    all_epoch_losses = []

    # Optimize the parameters
    for epoch in range(n_epochs):
        current_epoch_losses = []
        print(f"Epoch #{epoch}:")

        for i in range(len(train)):
            # Reset the gradients
            optimizer.zero_grad()

            model_output = model.forward(train[i])

            # Calculate the loss
            loss = ssim_loss(model_output["input_image"], model_output["output_image"])

            # Print the loss
            if i % 5 == 0:
                print(f"Step: {i}, Loss: {loss}")

            # Backpropagate the loss
            loss.requires_grad = True
            loss.backward(retain_graph=True)

            # Update the parameters
            optimizer.step()

            # Add the loss to the list of losses
            current_epoch_losses.append(loss)

        all_epoch_losses.append(current_epoch_losses)

    return all_epoch_losses


# Evaluate the gpt part of the model
def evaluate_gpt(model: DetectronGPTDiffusion, dataset_size=None):
    # Get the dataset
    dataset = get_image_filenames(
        "/content/drive/Shareddrives/COM SCI 263/Final Project/Data/COCO/val2017"
    )[:dataset_size]
    # We use run the model one time for each image in the dataset
    # (Filtered to only n_steps images)
    train, test = train_test_split(dataset, test_size=0.2, random_state=42)

    all_losses = []

    # Optimize the parameters
    for i in range(len(test)):
        model_output = model.forward(test[i])
        # Calculate the loss
        loss = ssim_loss(model_output["input_image"], model_output["output_image"])
        # Print the loss
        print(f"Step: {i}, Loss: {loss}")
        # Add the loss to the list of losses
        all_losses.append(loss)

    # Calculate the average loss
    average_loss = sum(all_losses) / len(all_losses)
    print(f"Average loss: {average_loss}")

    return all_losses


# Visualize the loss over time
def visualize_loss(all_epoch_losses):
    # Visualize the loss
    plt.plot(all_epoch_losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()


pipeline = DetectronGPTDiffusion()

total_losses = optimize_gpt(pipeline, n_epochs=2, dataset_size=10)
total_losses = [[loss.detach().numpy() for loss in epoch_loss] for epoch_loss in total_losses]

for epoch_losses in total_losses:
    visualize_loss(epoch_losses)

eval_losses = evaluate_gpt(pipeline, dataset_size=10)
eval_losses = [loss.detach().numpy() for loss in eval_losses] 

visualize_loss(eval_losses)