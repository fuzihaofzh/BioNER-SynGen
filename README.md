# <div align="center">Biomedical Named Entity Recognition via Dictionary-based Synonym Generalization</div>
<div align="center"><b>Zihao Fu,<sup>1</sup> Yixuan Su,<sup>1,2</sup> Zaiqiao Meng,<sup>3,1</sup> Nigel Collier<sup>1</sup></b></div>

<div align="center">
<sup>1</sup>Language Technology Lab, University of Cambridge<br>
<sup>2</sup>Cohere<br>
<sup>3</sup>School of Computing Science, University of Glasgow
</div>

[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/pdf/2305.13066.pdf)
[![YouTube Video](https://img.shields.io/badge/YouTube-Video-red.svg)](https://www.youtube.com/watch?v=nJnV_TcOAQA)
[![Slides](https://img.shields.io/badge/View-Slides-green.svg)](https://github.com/fuzihaofzh/BioNER-SynGen/blob/main/BioNER-slides.pdf)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)


## Introduction
BioNER-SynGen is a cutting-edge Synonym Generalization (SynGen) framework designed to enhance the task of biomedical named entity recognition. This work aims to solve the synonym generalization problem associated with dictionary-based methods, which often miss out on recognizing biomedical concept synonyms not listed in their respective dictionaries.

Through span-based predictions and the incorporation of two pivotal regularization terms, the SynGen framework sets a new standard in the field:

- **Synonym Distance Regularizer**: Ensures concepts closely related in meaning are also closely related in the model's internal representation.
  
- **Noise Perturbation Regularizer**: Adds a degree of noise resistance to the model, ensuring more robust synonym detection.

<img width="500" src="https://github.com/fuzihaofzh/BioNER-SynGen/assets/1419566/af77dc1b-7b04-41b6-aca9-778694d26ecd" alt="Your Image Description">


## Install 

```bash
git clone https://github.com/fuzihaofzh/BioNER-SynGen.git
cd BioNER-SynGen
pip install -r requirements.txt
./scripts/setup.sh
```

## Run the Experiments

We have conducted our experiments on 6 datasets including NCBI, BC5CDR-D, BC5CDR-D, BC4CHEMD, Species-800, and LINNAEUS. To replicate our experiments, please run each corresponding script. For example, to run SynGen on the NCBI dataset:

```bash
./scripts/run_ncbi.sh
```

You should expect output similar to the following:

```
Evaluating ncbi__edge,pos=umlsdi,neg=sap,ppreg,pndreg=0.1,ptm=pubmedbert
{
  "P": 0.685,
  "R": 0.631,
  "F1": 0.657
}
```

Do note, the scores might differ slightly from the results in the paper due to:

1. The inherent randomness when training neural networks.
2. We run each experiment 10 times with different random seeds to mitigate this randomness and report the average score.

## Use Your Own Dictionay
Kindly create a dictionary file that matches the format of "datasets/umls/umlsdi.txt". Once done, replace the original file with your newly-created dictionary and execute the "scripts/run_ncbi.sh" script.

## Cite
If you find our research and code beneficial, please consider citing our work:

```bibtex
@inproceedings{fu2023biomedical,
  title={Biomedical Named Entity Recognition via Dictionary-based Synonym Generalization},
  author={Fu, Zihao and Su, Yixuan and Meng, Zaiqiao and Collier, Nigel},
  booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2023}
}
```

## License
This project is licensed under the MIT License - refer to the [LICENSE](LICENSE) file for details.

## Support
For questions, issues, or any other form of communication, please reach out to the authors or open an issue on this repository.

---

## Acknowledgments

We would like to express our gratitude to the creators and contributors of the following projects, which have played an instrumental role in our research:

- [SapBERT from Cambridge LTL](https://github.com/cambridgeltl/sapbert)

- [PURE from Princeton NLP](https://github.com/princeton-nlp/PURE) 
