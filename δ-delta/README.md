# δ-delta
### δ-delta is a LM that is trained off of basic English and exclusively influenced directly off of 4chan

# Prerequisites
## Dependencies
Delta requires multiple python libraries which include:
- requests
- beautifulsoup4
- urllib3
- rich (optional but recomended)
```
python -m pip install requests beautifulsoup4 urllib3 rich
```
# Info & Help
## Setup
To setup Delta you first need to run train.py (this manages the whole model too, soon to be renamed and integrated with the ACOpsis shell) you can do this via 
```
python train.py train <arguments>
```
or
```
python3 train.py train <arguments>
```
depending on your setup.
## Model selection
Delta allows selection of external models via Hugging Face, you can do this by running:
```
python train.py train --models <link to model>
```
although this usually requires
```
--trust-remote-code
```
as an argument appended to the run
## Options
Delta is a highly customizable model thus has many options including:
- Collect (harvest 4Chan threads)
- Train (run training)
- Generate (generate output)
- Stats (corpus info)
### Refrence
```
    python train.py collect --boards g <threads> --max-threads <thread max count>
    python train.py train   --epochs <epoch count>
    python train.py generate --prompt "<prompt>"
    python train.py generate -i          # Chatbot like back-and-forth prompt shell
    python train.py stats                # corpus summary
    python train.py train --dry-run      # preview config + memory, no training
    python train.py train --model <HF repo link> # Use a public HF model
```
