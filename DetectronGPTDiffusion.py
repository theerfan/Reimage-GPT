import os

import cv2
import detectron2
import numpy as np
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from diffusers import DiffusionPipeline

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import pathlib


class DetectronGPTDiffusion(nn.Module):
    def __init__(self):
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
        self.predictor = DefaultPredictor(self.cfg)

        # GPT-J parts
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", cache_dir=pathlib.Path("cache").resolve()
        ).to(self.device)

        # Diffusion parts
        self.diffusion_model = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ).to(self.device)

    def partition_image(self, image_dir: str = "input.jpg"):
        im = cv2.imread(image_dir)
        outputs = self.predictor(im)
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

        return [boxes, scores, classes, labels, keypoints]

    def gpt_post_process(self, output_sequences):
        predictions = []
        generated_sequences = []

        max_repeat = 2

        # decode prediction
        for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
            generated_sequence = generated_sequence.tolist()
            text = self.tokenizer.decode(
                generated_sequence,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            generated_sequences.append(text.strip())

        for i, g in enumerate(generated_sequences):
            res = str(g).replace("\n\n\n", "\n").replace("\n\n", "\n")
            lines = res.split("\n")
            # # print(lines)
            # i = max_repeat
            # while i != len(lines):
            #   remove_count = 0
            #   for index in range(0, max_repeat):
            #     # print(i - index - 1, i - index)
            #     if lines[i - index - 1] == lines[i - index]:
            #       remove_count += 1
            #   if remove_count == max_repeat:
            #     lines.pop(i)
            #     i -= 1
            #   else:
            #     i += 1
            predictions.append("\n".join(lines))

        return predictions

    def gpt_generate(self, prompt):
        num_sequences = 10
        # Generation settings:
        min_length = 100
        max_length = 160
        temperature = 1  # min:0, max:3, step:0.01
        top_p = 0.95  #  min:0, max:1, step:0.01
        top_k = 50
        # This is very critical for GPT-2
        repetition_penalty = 1.0

        prompt_full = self.tokenizer(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        encoded_prompt = prompt_full.input_ids
        encoded_prompt = encoded_prompt.to(self.device)
        # max_length = tokenizer.model_max_length
        # prediction
        output_sequences = self.gpt.generate(
            input_ids=encoded_prompt,
            max_length=max_length,
            attention_mask=prompt_full.attention_mask,
            min_length=min_length,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            do_sample=True,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_sequences,
        )

        return self.gpt_post_process(output_sequences)

    def generate_image(self, prompt):
        return self.diffusion_model(prompt).images


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
