
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import random
import string
from tqdm import tqdm

# def generate_sentence():
#     return ''.join(random.choices(string.ascii_uppercase, k=16))


def generate_sentences(num_sentences, pattern='AABBCCDDDDEEEEFF'):
    sentences = []
    all_chars = string.ascii_uppercase
    for _ in range(num_sentences):
        # Generate a random start index for the rotation
        start_index = all_chars.index(random.choice(string.ascii_uppercase))

        pattern = ""
        pattern = all_chars[ start_index]
        pattern+=all_chars[ start_index]
        pattern+=all_chars[ (start_index +1) % len(all_chars)]
        pattern+=all_chars[ (start_index +1) % len(all_chars)]
        pattern+=all_chars[ (start_index +2) % len(all_chars)]
        pattern+=all_chars[ (start_index +2) % len(all_chars)]
        pattern+=all_chars[ (start_index +3) % len(all_chars)]
        pattern+=all_chars[ (start_index +3) % len(all_chars)]
        pattern+=all_chars[ (start_index +3) % len(all_chars)]    
        pattern+=all_chars[ (start_index +3) % len(all_chars)]
        pattern+=all_chars[ (start_index +4) % len(all_chars)]
        pattern+=all_chars[ (start_index +4) % len(all_chars)]
        pattern+=all_chars[ (start_index +4) % len(all_chars)]
        pattern+=all_chars[ (start_index +4) % len(all_chars)]
        pattern+=all_chars[ (start_index +5) % len(all_chars)]
        pattern+=all_chars[ (start_index +5) % len(all_chars)]

        # for i in range(3):
        #     pattern += all_chars[(start_index + i) % len(all_chars)] * 2

        
        # Generate the rotated pattern
        start_index = np.random.randint(len(pattern))
        rotated_pattern = pattern[start_index:] + pattern[:start_index]

        
        
        # If the rotated pattern starts with a character other than 'A', continue the pattern in sequence
        if rotated_pattern[0] != 'A':
            next_char_index = all_chars.index(rotated_pattern[0]) + 1
            next_char = all_chars[next_char_index % len(all_chars)]
            repeat_count = pattern.count(next_char)
            rotated_pattern += next_char * (len(pattern) - len(rotated_pattern))

        sentences.append(rotated_pattern)

    return sentences

# training_sentences = [generate_sentence() for _ in range(10000)]
# validation_sentences = [generate_sentence() for _ in range(2000)]
# test_sentences = [generate_sentence() for _ in range(2000)]
training_sentences = generate_sentences(10000)
validation_sentences = generate_sentences(2000)
test_sentences = generate_sentences(2000)

with open('training_data.txt', 'w') as f:
    for sentence in training_sentences:
        f.write(sentence + '\n')

with open('validation_data.txt', 'w') as f:
    for sentence in validation_sentences:
        f.write(sentence + '\n')

with open('test_data.txt', 'w') as f:
    for sentence in test_sentences:
        f.write(sentence + '\n')

# Check if CUDA is available and set PyTorch to use GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

class SentenceDataset(Dataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            # read all the lines in the file and store them in the sentences list without the newline character
            self.sentences = f.readlines()
            self.sentences = [sentence.strip() for sentence in self.sentences]

        # Create a vocabulary of unique characters
        self.vocab = list(set(''.join(self.sentences)))
        # Add '*' to the vocabulary
        if '*' not in self.vocab:
            self.vocab.append('*')
        # print(f'Vocabulary: {self.vocab}')
        # Create a dictionary mapping each character to a unique index
        self.char_to_index = {char: index for index, char in enumerate(self.vocab)}
        self.int_to_char = {i: ch for ch, i in self.char_to_index.items()}
    
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Convert the sentence into numerical form
        numerical_sentence = [self.char_to_index[char] for char in sentence]

        # Create a copy of the numerical sentence for the target
        target = numerical_sentence.copy()

        # Randomly choose the percentage of characters to mask (between 10% and 40%)
        mask_percentage = np.random.uniform(0.1, 0.4)
        num_chars_to_mask = int(mask_percentage * len(numerical_sentence))

        # Randomly choose the indices of the characters to mask
        mask_indices = np.random.choice(len(numerical_sentence), num_chars_to_mask, replace=False)

        # Mask the chosen characters in the input
        for index in mask_indices:
            numerical_sentence[index] = self.char_to_index['*']



        return torch.tensor(numerical_sentence), torch.tensor(target)

class SentencePredictionModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout_rate=0.1):
        super(SentencePredictionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.transformer = nn.Transformer(hidden_size, num_encoder_layers=6, num_decoder_layers=6, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)  # Apply dropout after embedding
        output = self.transformer(embedded, embedded)
        output = self.fc(output)
        # output = self.norm(output)  # Apply normalization after fully connected layer

        return output

# Hyperparameters
vocab_size = 35  # Assuming ASCII characters
hidden_size = 256 
batch_size = 256
num_epochs = 50
learning_rate = 0.001

# Load the dataset
train_dataset = SentenceDataset('training_data.txt')
valid_dataset = SentenceDataset('validation_data.txt')
test_dataset = SentenceDataset('test_data.txt')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = SentencePredictionModel(vocab_size, hidden_size)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0
    total_train_samples = 0
    train_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch+1}/{num_epochs}")

    for batch in train_bar:
        sentences, targets = batch
        optimizer.zero_grad()
        input = sentences.to(device)  # Input sentence with masked characters
        target = targets.to(device)  # Target sentence with missing characters to predict
        output = model(input)
        loss = criterion(output.view(-1, vocab_size), target.view(-1))
        loss.backward()
        optimizer.step()
    
        total_train_loss += loss.item() * input.size(0)
        total_train_samples += input.size(0)
        train_loss = total_train_loss / total_train_samples
        train_bar.set_postfix({'loss': train_loss})

    model.eval()
    total_valid_loss = 0
    total_valid_samples = 0
    valid_bar = tqdm(valid_dataloader, desc=f"Validating epoch {epoch+1}/{num_epochs}")
  
    with torch.no_grad():
        for batch in valid_bar:
            sentences, targets = batch
            input = sentences.to(device)  # Input sentence with masked characters
            target = targets.to(device)  # Target sentence with missing characters to predict
            output = model(input)
            loss = criterion(output.view(-1, vocab_size), target.view(-1))

            total_valid_loss += loss.item() * input.size(0)
            total_valid_samples += input.size(0)
            valid_loss = total_valid_loss / total_valid_samples
            valid_bar.set_postfix({'loss': valid_loss})
            
                   
        if valid_loss < 0.2:
            # Save checkpoint
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


def test_model(model, test_dataloader):
    model.eval()
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            sentences, targets = batch
            sentences = sentences.to(device)  
            targets = targets.to(device) 

            output = model(sentences)
            predicted = output.argmax(dim=2)

            for i in range(len(sentences)):
                sentence = sentences[i]
                target = targets[i]
                prediction = predicted[i]
    
                correct_predictions += (prediction == target).sum().item()
               
                sentence = ''.join(test_dataloader.dataset.int_to_char[int(i)] for i in sentences[i])
                target = ''.join(test_dataloader.dataset.int_to_char[int(i)] for i in targets[i])
                prediction = ''.join(test_dataloader.dataset.int_to_char[int(i)] for i in predicted[i])

                total_predictions += len(sentence)
            
            print(f'\n Input: {sentence} Target: {target} Predicted: {prediction}')

    accuracy = correct_predictions / total_predictions
    print(f'Test Accuracy: {accuracy * 100}%')
    
# Test the model
test_model(model, test_dataloader)