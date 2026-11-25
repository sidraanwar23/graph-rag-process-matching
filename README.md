<a id="readme-top"></a>

<h3 align="center">Graph Context for LLM-based Process Model Matching</h3>

<p align="center">
  Can an LLM match business process activities better if it can see the steps around them?
  <br />
  <br />
  <a href="https://colab.research.google.com/github/sidraanwar23/graph-rag-process-matching/blob/main/01_data_and_baselines.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open notebook 01 in Colab"></a>
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
</p>

<details>
  <summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-project">About the project</a>
      <ul><li><a href="#built-with">Built with</a></li></ul>
    </li>
    <li><a href="#data">Data</a></li>
    <li><a href="#what-i-compared">What I compared</a></li>
    <li><a href="#results">Results</a></li>
    <li><a href="#getting-started">Getting started</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About the project

When two organizations merge, someone has to work out which steps in their business processes are actually the same. That sounds easy until you look at the labels. "Notify students to collect cards" and "Inform students about card collection" describe the same thing, while two labels that look alike can describe different work.

My MS thesis tackled this with text alone: I used T5 to expand short activity labels into fuller descriptions and then classified pairs. That worked well, but it ignored something obvious. A process model is a graph, and every activity sits between other activities.

After reading *The Reasoning Bottleneck in Graph-RAG* (Zarrinkia, Srinivasan and Thomo, 2026), I wanted to test a simple idea: give the LLM each activity's neighbours (what comes before, what comes after, and who does it) and see whether matching improves.

### Built with

* Python, PyTorch, scikit-learn, NetworkX
* Hugging Face Transformers: FLAN-T5, all-MiniLM-L6-v2, Qwen2.5-7B-Instruct (4-bit)
* Google Colab (T4 GPU)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Data

I used the University Admission set from the [Process Model Matching Contest 2015](https://ai.wu.ac.at/emisa2015/contest.php). It has 9 BPMN models describing how German universities handle Master's applications, 36 model pairs, and expert alignments that say which activities match.

Each model is turned into a directed graph. Activities are nodes, the lane or pool gives the actor, and an edge links an activity to the next one (gateways and events in between are skipped).

## What I compared

| Method | What the model sees |
|---|---|
| Feature-based classifier | FLAN-T5 descriptions, string and embedding similarity, Gradient Boosting |
| LLM, labels only | just the two activity labels |
| LLM + graph context | labels, up to 3 steps before and after, and the actor |
| LLM + graph context + checklist | the same, plus a short "check step by step" prompt |
| Hybrid | the feature-based classifier with the three LLM scores added |

To keep things fair, anything that is trained or tuned uses 6-fold cross-validation grouped by model pair, so the model never sees the test pair during training. The LLM only scores the top 3 candidates per activity. Those candidates contain 82% of the true matches, and everything else counts as "no match".

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Results

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Feature-based classifier | 0.791 | 0.702 | **0.744** |
| Hybrid | 0.703 | 0.730 | 0.716 |
| LLM, labels only | 0.529 | 0.663 | 0.589 |
| LLM + graph context + checklist | 0.764 | 0.309 | 0.440 |
| LLM + graph context | 0.787 | 0.208 | 0.329 |

*LLM rows use P(Yes) ≥ 0.5.*

The result surprised me. Graph context made the LLM much more precise (0.53 to 0.79), but it also made it far too cautious, and recall dropped to 0.21. My first look at the errors points to one possible reason: universities may do equivalent steps in a different order, so two matching activities can have quite different neighbours, and the model takes that as a sign they are different. The checklist prompt won back part of the loss.

This is a small benchmark, so I treat these as early results rather than conclusions. The obvious next question is when graph context should be used at all.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting started

Both notebooks run on Google Colab with a free T4 GPU. Nothing needs to be installed locally.

1. Open `01_data_and_baselines.ipynb` in Colab (badge above) and choose **Runtime → Change runtime type → T4 GPU**.
2. Run all cells. The notebook downloads the dataset, builds the graphs, runs both baselines and saves `outputs.zip`.
3. Open `02_graph_context.ipynb`, upload `outputs.zip` to the session, and run all cells.

Each notebook takes roughly 30 to 45 minutes, mostly spent on LLM inference.

| Notebook | What it does |
|---|---|
| `01_data_and_baselines.ipynb` | data preparation, process graphs, gold alignments, the two baselines |
| `02_graph_context.ipynb` | graph-context prompts, evaluation, decisions that changed, worked examples |

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

- [x] Rebuild my thesis approach on a public benchmark
- [x] LLM baseline with labels only
- [x] Graph context and a structured checklist prompt
- [ ] Use graph context only for uncertain pairs
- [ ] Rank neighbours by relevance before adding them to the prompt
- [ ] Repeat on the Birth Registration dataset
- [ ] Add neighbour-similarity features to the feature-based classifier

## License

Distributed under the MIT License. See `LICENSE` for details.

## Contact

Sidra Anwar · Sidraanwar050@gmail.com

Project link: [github.com/sidraanwar23/graph-rag-process-matching](https://github.com/sidraanwar23/graph-rag-process-matching)

## Acknowledgments

* Y. Zarrinkia, V. Srinivasan and A. Thomo. [The Reasoning Bottleneck in Graph-RAG: Structured Prompting and Context Compression for Multi-Hop QA](https://arxiv.org/abs/2603.14045). arXiv, 2026.
* The organizers of the [Process Model Matching Contest 2015](https://ai.wu.ac.at/emisa2015/contest.php) for the dataset and gold standard.
* My MS supervisors at COMSATS University Islamabad, Lahore Campus, Dr. Abid Sohail Bhutta and Dr. Farooq Ahmad.
* README layout adapted from [Best-README-Template](https://github.com/othneildrew/Best-README-Template).

<p align="right">(<a href="#readme-top">back to top</a>)</p>
