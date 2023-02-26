# Multimodal Chain-of-Thought Reasoning in Language Models

<h5 align="center"><i>"Imagine learning a textbook without figures or tables."</i></h5>

Multimodal-CoT incorporates vision features in a decoupled training framework. The framework consists of two training stages: (i) rationale generation and (ii) answer inference. Both stages share the same model architecture but differ in the input and output.

## Requirements

Install all required python dependencies:

```
pip install -r requirements.txt
```

## Datasets

Download the dataset from the following repository:

```
https://github.com/lupantech/ScienceQA/tree/main/data
```

Download the extracted vision features from [vision_features](https://drive.google.com/file/d/13B0hc_F_45-UlqPLKSgRz-ALtFQ8kIJr/view?usp=share_link) and unzip the files under `vision_features`

## Instructions

### Training | Inference

checkout [shell cmd](readme/usage.txt)

```

## Citing MM-CoT

```
@article{zhang2023multicot,
  title={Multimodal Chain-of-Thought Reasoning in Language Models},
  author={Zhang, Zhuosheng and Zhang, Aston and Li, Mu and Zhao, Hai and Karypis, George and Smola, Alex},
  journal={arXiv preprint arXiv:2302.00923},
  year={2023}
}
```

## License

This project is licensed under the Apache-2.0 License.

## Acknowledgement

Part of our codes are adapted from [ScienceQA](https://github.com/lupantech/ScienceQA) and [Transformers](https://github.com/huggingface/transformers).

We thank [Pan Lu](https://lupantech.github.io/) for providing parameter size for ScienceQA baselines.
