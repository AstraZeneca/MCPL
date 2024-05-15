# Multi-Concept-Dataset

This is the dataset we collected for our Multi-Concepts learning research, if you found this dataset useful, plese [cite](https://scholar.googleusercontent.com/scholar.bib?q=info:1P28Cg5FBCYJ:scholar.google.com/&output=citation&scisdr=ClEjyECxEMG0_68ingg:AFWwaeYAAAAAZgIkhgj_fg-Gx2UinlXEQPSzW-U&scisig=AFWwaeYAAAAAZgIkhtDxly2dHs3GWoB6AhLPzk8&scisf=4&ct=citation&cd=-1&hl=en) our work. 

### An Image is Worth Multiple Words: Discovering Object Level Concepts using Multi-Concept Prompt Learning

<a href="https://astrazeneca.github.io/mcpl.github.io/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2310.12274"><img src="https://img.shields.io/badge/arXiv-2305.16311-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch->=1.10.0-Red?logo=pytorch"></a>
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/papers/2310.12274)


## Dataset
[Reference and Licenses](references_and_licenses.txt) 

We generate and collected a [Multi-Concept-Dataset](.) including a total of ~ 1400 images and masked objects/concepts as follows

  /  (370 images)
  /natural_2_concepts  
  /natural_345_concepts  
  /real_natural_concepts

| Data file name | Size | # of images |
| --- | --- | ---: |
| [medical_2_concepts](medical_2_concepts/) | 2.5M | 370 |
| [natural_2_concepts](natural_2_concepts/) | 36M | 415 |
| [natural_345_concepts](natural_345_concepts/) | 13M | 525 |
| [real_natural_concepts](real_natural_concepts/) | 5.6M | 137 |

## Citation

If you make use of our work, please cite our paper:

```
@article{jin2023image,
  title={An image is worth multiple words: Learning object level concepts using multi-concept prompt learning},
  author={Jin, Chen and Tanno, Ryutaro and Saseendran, Amrutha and Diethe, Tom and Teare, Philip},
  journal={arXiv preprint arXiv:2310.12274},
  year={2023}
}
```
