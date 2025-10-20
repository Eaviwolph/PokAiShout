# PokAiShout
PokAiShout is an AI-powered application that generates custom shout audio clips based on user input text.

## Features
Scrap Pokepedia for sound clips of Pokemon shouts.
```bash
python scrap.py
```

Train a custom AI model to generate new shout audio clips.
```bash
python ai.py --train --data-dir ./audios --epochs 30 --samples 1007
```

Generate new shout audio clips using the trained AI model.
```bash
python3 ai.py --model name_of_your_ai.h5 --name "NameYouWant"
```
