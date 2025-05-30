<h1 align="center">🎓 UChicago Playground (2023-2025)</h1>
<p align="center"><em>"Efforts, Partial and Unsatisfactory"</em> at Ryerson Floor 2, UChicago</p>

My economics professor in college once told me that UChicago is a sacred place for academic pursuit. Later, my first advisor at UChicago said he hoped that something of UChicago would stay with me. Looking back on my three-year journey—one year as an undergrad RA at Chicago Booth and two years of master's study—this has been an intensive but deeply rewarding experience. It transformed me from a traditional data-driven machine learning researcher into a human-centered NLP practitioner.

At this turning point, I want to take a moment to reflect and summarize my experiences in this repository. Beyond formal publications, many projects here—though small or toyish—capture some key “aha” moments throughout this journey.

Due to time constraints, I've mainly included **structured outputs (e.g., reports, code, posters) related to NLP topics**. But I've also gained so much solid training in **proof-based statistics and optimization theory** and hope to eventually convert my handwritten notes into LaTeX versions sometime—though I still enjoy the charm of writing by hand! 🤡

### 📋 Coding-intensive Course Summary

| Course Number         | Course Name                                 | Keywords                                      | Output                 |
|-----------------------|---------------------------------------------|-----------------------------------------------|----------------------|
| BUSN 32200/32810 | [Artificial Intelligence](#ta-artificial-intelligence)                     | Comprehensive Survey of AI: From Academia to Industry to Society          | TA (Creating Course Materials)          |
| CMSC 35200            | [Deep Learning Systems](#deep-learning-systems)                    | Systems, Sparsification, Inference (Ollama, vLLM), Evaluation            | Pilot Project (Clinical NLP)       |
| CMSC 35401            | [Human-Centered AI](#human-centered-ai)                           | Alignment, Weak-to-Strong Generalization             | Toy Project (Financial NLP)       |
| DATA 37784            | [Representation Learning in Machine Learning](#representation-learning-in-machine-learning)| Causal Representation Learning, Model Editing| Toy poject (Interpretability)      |
| CMSC 35440            | [Machine Learning in Biology and Medicine](#machine-learning-in-biology-and-medicine)                  |    -    | Coding Assignments (Python)       |
| STAT 31900            | [Introduction to Causal Inference](#introduction-to-causal-inference)            |    -     | Coding Assignments (R)        |
| FINM 32950            | [Introduction to High Performance Computing (HPC)](#introduction-to-high-performance-computing-hpc)   |       -       | Coding Assignments (C++)       |

---

## (TA) Artificial Intelligence
🔗 [📚 My Reading List](https://github.com/YuyangJ0/UChicago-Playground/tree/main/BUSN_32200) 

- **Instructor:** Dacheng Xiu, Booth School of Business  
- **Timeline:** 2023 Summer (Volunteer), 2024 Summer (TA), 2025 Winter (TA)
- **Comment:** I have been honored to be part of Professor Dacheng Xiu's teaching assistant team since 2023, contributing to the establishment of the first Artificial Intelligence course at Chicago Booth and supporting its growth from an initial concept into a comprehensive 8-lecture series. My initial project in 2023 involved **creating a cartoon-style text-to-video syllabus**. For this, I utilized OpenAI’s Text-to-Speech (TTS) model to convert text into speech audio, trained a so-vits-svc model to adapt this synthesized audio to closely match the instructor’s voice, and produced animated visuals of the instructor using Midjourney and D-ID for synchronized audio-video presentations. My primary role has since focused on **developing extensive course materials** for six new lectures. For the summer course, I created business-oriented case studies and conducted thorough literature reviews for two new lectures: Generative AI and the Social Impact of AI. In the winter course, I analyzed and summarized key model examples, training pipelines, and underlying mechanisms drawn from academic research and technical reports for four additional lectures: Dataification, Large Language Models, Reasoning Models, and Vision-Language Models.



## Deep Learning Systems  
🔗 [🩻 Poster](https://github.com/YuyangJ0/UChicago-Playground/blob/main/CMSC_35200/poster_24x36.pdf) 🔗 [📄 Report](https://github.com/YuyangJ0/UChicago-Playground/blob/main/CMSC_35200/Evaluator_report_20241212.pdf) 🔗 [💻 Code](https://github.com/YuyangJ0/UChicago-Playground/tree/main/CMSC_35200/code)

- **Instructor:** Rick Stevens, Department of Computer Science 
- **Timeline:** 2024 Autumn 
- **Title:** Exploring the Potential of LLMs as Radiology Report Evaluator 
- **Abstract:** The rise of large language models (LLMs) has opened new possibilities in medical AI, particularly in evaluating radiology reports. This study explores the potential of LLMs as radiology report evaluators. Current evaluation frameworks for radiology report generation are often shallow and limited, leading to inconclusive results about AI systems' capabilities. To address this, we propose a comprehensive pipeline for evaluating generated reports using LLMs, grounded in tasks such as label extraction and clarity scoring. Our experiments demonstrate that LLMs consistently outperform state-of-the-art (SOTA) labelers in labeling accuracy and show promise in clarity evaluation, particularly when leveraging agentic collaboration like majority voting. However, challenges remain, including mixed results with prompt engineering strategies, the limitations of multi-step prompting, and failure to fully align with radiologists' preferences for filtering more readable reports. This work represents a step forward in developing a robust, LLM-based evaluation framework, ultimately aimed at enhancing the quality and reliability of radiology report generation pipelines.
- **Comment:** This pilot project led to my evaluation methodology paper CLEAR ([preprint](https://arxiv.org/abs/2505.16325)), along with the full code and a curated dataset. 💪❤️‍🔥


## Human-Centered AI  
🔗 [📄 Report](https://github.com/YuyangJ0/UChicago-Playground/blob/main/CMSC_35401/CMSC_35401_Final_report.pdf) 🔗 [💻 Code]

- **Instructor:** Chenhao Tan, Department of Computer Science 
- **Timeline:** 2024 Spring 
- **Title:** Weak-to-Strong Generalization on Financial News Summarization
- **Abstract:** In this project, we investigate the applicability of the weak-to-strong generalization framework in the context of financial news summarization. We constructed a dataset of 2,000 financial news records and assessed the performance of OpenAI's framework from lexical and pragmatic viewpoints. Our findings indicate that: (1) the weak-to-strong generalization does not hold in the context of financial news summarization for the model classes tested; (2) while the summarization models like BART series more closely align with the ground truth compared to the text generation model LLaMA-1, they are less effective in producing financially informative summaries.
- **Comment:** This toy project tests OpenAI's [*Weak-to-Strong Generalization*](https://openai.com/index/weak-to-strong-generalization/) framework on a domain-specific task-**Financial News Summarization**.


## Representation Learning in Machine Learning  
🔗 [📄 Proposal](https://github.com/YuyangJ0/UChicago-Playground/blob/main/DATA_37784/report.pdf) 🔗 [💻 Code](https://github.com/YuyangJ0/UChicago-Playground/tree/main/DATA_37784/code)

- **Instructor:** Victor Veitch, Department of Statistics  
- **Timeline:** 2024 Spring 
- **Title:** Locating Domain-specific Facts in Generative Pre-training Transformer (GPT)
- **Abstract:** This proposal explores two key questions: (1) How can we refine the definition of "factual" associations by distinguishing domain-specific facts, and does the location of such knowledge within GPT vary accordingly? (2) While fine-tuning is commonly used to develop domain-specific models, how does it compare to domain-specific model editing in terms of effectiveness and outcomes? (the answer for the second one is a direct sad "NO" when I discussed this with professor 😔)
- **Comment:** This toy project was inspired by my presented paper, [*Locating and Editing Factual Associations in GPT*](https://rome.baulab.info/), on the course. The paper introduces an interesting approach using **Causal Tracing** to identify neurons that store factual associations. Based on location findings, the authors proposed the **Rank-One Model Editing (ROME)** method to directly edit the model's memory of specific facts.


## Machine Learning in Biology and Medicine  
🔗 [📝 Assignments](https://github.com/YuyangJ0/UChicago-Playground/tree/main/CMSC_35440)

- **Instructor:** Robert Grossman, Departments of Medicine & Computer Science  
- **Skill Sets:** Clinical NLP (Embeddings, Attention, Transformer, LLM, RAG, Agents)
- **Status:** A 💯


## Introduction to Causal Inference  
🔗 [📝 Assignments](https://github.com/YuyangJ0/UChicago-Playground/tree/main/STAT_31900)

- **Instructor:** Guanglei Hong, Department of Comparative Human Development  
- **Skill Sets:**  Causal Inference (Experimental Designs, Propensity Score-Based Methods (IPTW, MMWS), Instrumental Variable (IV) Method, Regression Discontinuity Design (RDD), Difference-in-Differences Analysis)
- **Status:** A 💯


## Introduction to High Performance Computing (HPC)  
🔗 [📝 Assignments]

- **Instructor:** Chanaka Liyanaarachchi, Department of Mathematics  
- **Skill Sets:**  Parallel and accelerator computing (Threads, OpenMP, Intel oneAPI, CUDA (GPGPU), advanced Python techniques), Pattern-based design for concurrency and parallelism, Linux Systems
- **Status:** On-going


## 🙋‍♀️ Q&A 

If you encounter any issues or have suggestions for improvement, feel free to open an issue in this repository. I'm always happy and open to discuss further!


## 📄 License & Attribution

© 2025 Yuyang Jiang. All rights reserved.

This repository contains original work. Unauthorized copying, reuse, or distribution is prohibited. If you wish to reference or use this material, please contact me at kjiang4work@gmail.com. Thanks for your understanding!

