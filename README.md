# deep-chess
Chess AI with neural network based move evaluation

## Data
The model was trained on evaluated lichess games https://database.lichess.org

## Data processing
### 0. Split pgn file into games
### 1. remove games without lichess evaluation ```python keep_eval_games.py```

## Data Generation

### 2. generate data ```python generate_data.py```

Note: Having up to 700,000 games (unsplitted pgn file of size 1.5gb approx) can land you at to 1.7 million data points

## Model training

### 3. ```python train_net.py```