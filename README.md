<a id="readme-top"></a>

<h3 align="center">Graph Context for LLM-based Process Model Matching</h3> <p align="center"> Can an LLM match business process activities better if it can see the steps around them? <br /> <br /> <a href="https://colab.research.google.com/github/sidraanwar23/graph-rag-process-matching/blob/main/01_data_and_baselines.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open notebook 01 in Colab"></a> <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"> <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"> </p> <details> <summary>Table of contents</summary> <ol> <li><a href="#about-the-project">About the project</a> <ul><li><a href="#built-with">Built with</a></li></ul> </li> <li><a href="#data">Data</a></li> <li><a href="#what-i-compared">What I compared</a></li> <li><a href="#results">Results</a></li> <li><a href="#why-graph-context-hurts">Why graph context hurts</a></li> <li><a href="#getting-started">Getting started</a></li> <li><a href="#roadmap">Roadmap</a></li> <li><a href="#license">License</a></li> <li><a href="#contact">Contact</a></li> </ol> </details>
About the project

When two organizations merge, someone has to work out which steps in their business processes are actually the same. That sounds easy until you look at the labels. "Notify students to collect cards" and "Inform students about card collection" describe the same thing, while two labels that look alike can describe different work.

My MS thesis tackled this with text alone: I used T5 to expand short activity labels into fuller descriptions and then classified pairs. That worked well, but it ignored something obvious. A process model is a graph, and every activity sits between other activities.

After reading The Reasoning Bottleneck in Graph-RAG (Zarrinkia, Srinivasan and Thomo, 2026), I wanted to test a simple idea: give the LLM each activity's neighbours (what comes before, what comes after, and who does it) and see whether matching improves. When it didn't, I went back to find out why.

Built with
Python, PyTorch, scikit-learn, NetworkX
Hugging Face Transformers: FLAN-T5, all-MiniLM-L6-v2, Qwen2.5-7B-Instruct (4-bit)
Google Colab (T4 GPU)
<p align="right">(<a href="#readme-top">back to top</a>)</p>
Data

I used the University Admission set from the Process Model Matching Contest 2015. It has 9 BPMN models describing how German universities handle Master's applications, 36 model pairs, 20,661 activity pairs and 178 expert-labelled matches.

Each model is turned into a directed graph. Activities are nodes, the lane or pool gives the actor, and an edge links an activity to the next one (gateways and events in between are skipped).

What I compared
Method	What the model sees
Feature-based classifier	FLAN-T5 descriptions, string and embedding similarity, Gradient Boosting
LLM, labels only	just the two activity labels
LLM + graph context	labels, up to 3 steps before and after, and the actor
LLM + graph context + checklist	the same, plus a short "check step by step" prompt
Hybrid	the feature-based classifier with the three LLM scores added

To keep things fair, anything that is trained or tuned uses 6-fold cross-validation grouped by model pair, so the model never sees the test pair during training. The LLM only scores the top 3 candidates per activity (4,244 pairs). Those candidates contain 82% of the true matches, and everything else counts as "no match".

<p align="right">(<a href="#readme-top">back to top</a>)</p>
Results
Method	Precision	Recall	F1
Feature-based classifier	0.791	0.702	0.744
Hybrid	0.703	0.730	0.716
LLM, labels only	0.529	0.663	0.589
LLM + graph context + checklist	0.764	0.309	0.440
LLM + graph context	0.787	0.208	0.329

LLM rows use P(Yes) ≥ 0.5.

The result surprised me. Graph context made the LLM much more precise (0.53 to 0.79), but it also made it far too cautious, and recall dropped to 0.21. The checklist prompt won back part of the loss.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
Why graph context hurts

Notebook 03 looks inside those failures. For 400 candidate pairs I removed one context element at a time (a step before, a step after, or the actor) and measured how much the model's P(Yes) changed.

One neighbour can flip a decision. Graph context made the model reject 64 true matches. In the clearest case, both universities had an activity called "Send letter of rejection". With labels only, P(Yes) was 1.00. With context it was 0.00, and almost all of that came from a single neighbouring step, "Evaluate" (−0.99). Hiding that one step brought the answer back.

The explanations are faithful. Removing the top-ranked element changed P(Yes) by 0.164 on average, against 0.047 for a random element, and flipped the decision 17.5% of the time against 4.3%.

Step order is not the whole story. I expected neighbours with no counterpart in the other process to do the damage. On the lost matches, though, neighbours with a counterpart pushed towards "no match" about as hard as those without (mean −0.117 vs −0.110). Context in general can mislead the model, not just mismatched steps.

Two repairs, neither using gold labels:

Method	Precision	Recall	F1
LLM, labels only	0.529	0.663	0.589
LLM + full context + checklist	0.764	0.309	0.440
LLM + neighbours with a counterpart only	0.705	0.483	0.573
Selective context (only when 0.3 ≤ P(Yes) ≤ 0.7)	0.551	0.640	0.592
Selective + neighbours with a counterpart only	0.553	0.646	0.596

Keeping only the neighbours that have a counterpart in the other process recovered 31 true matches and lifted recall from 0.31 to 0.48 while keeping precision at 0.71. Selective context barely changed anything, for an interesting reason: only 22 of 4,244 candidates fell in the uncertain band. The model is almost always fully confident, even when it is wrong.

The feature-based classifier is still the strongest single method. This is a small benchmark, so I treat all of this as early evidence rather than a conclusion.

<p align="right">(<a href="#readme-top">back to top</a>)</p>
Getting started

All three notebooks run on Google Colab with a free T4 GPU. Nothing needs to be installed locally.

Open 01_data_and_baselines.ipynb in Colab (badge above) and choose Runtime → Change runtime type → T4 GPU.
Run all cells. The notebook downloads the dataset, builds the graphs, runs both baselines and saves outputs.zip.
Open 02_graph_context.ipynb, upload outputs.zip to the session, and run all cells. It saves outputs_graph_context.zip.
Open 03_explaining_graph_context.ipynb, upload both zip files, and run all cells.

Notebooks 01 and 02 take roughly 30 to 45 minutes each, and notebook 03 about an hour, mostly spent on LLM inference.

Notebook	What it does
01_data_and_baselines.ipynb	data preparation, process graphs, gold alignments, the two baselines
02_graph_context.ipynb	graph-context prompts, evaluation, decisions that changed, worked examples
03_explaining_graph_context.ipynb	occlusion attribution, faithfulness test, order-robust and selective context
<p align="right">(<a href="#readme-top">back to top</a>)</p>
Roadmap
[x] Rebuild my thesis approach on a public benchmark
[x] LLM baseline with labels only
[x] Graph context and a structured checklist prompt
[x] Find which context elements mislead the model, and check that the explanations are faithful
[x] Use graph context only for uncertain pairs
[ ] Calibrate the LLM's confidence so selective context has pairs to act on
[ ] Rank neighbours by relevance before adding them to the prompt
[ ] Repeat on the Birth Registration dataset
[ ] Add neighbour-similarity features to the feature-based classifier
License

Distributed under the MIT License. See LICENSE for details.

Contact

Sidra Anwar · sidraanwar050@gmail.com

Project link: github.com/sidraanwar23/graph-rag-process-matching

<p align="right">(<a href="#readme-top">back to top</a>)</p>
