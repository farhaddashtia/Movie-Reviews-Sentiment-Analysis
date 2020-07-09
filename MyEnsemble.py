# Install packages
import torch
from torchtext import data
from torchtext import datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import spacy
import time
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from torch.utils.data import RandomSampler

import nltk
import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
from torchtext import data
from torchtext import datasets


#Load English language
nlp = spacy.load('en')

# Initiate class instances with tokenizers
TEXT = data.Field(tokenize = 'spacy', batch_first = True,pad_first = True,fix_length=1300)
LABEL = data.LabelField(dtype = torch.float)
# Load data from torchtext (identical to what we have in Kaggle)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
#-------------------------------------------
train_data, valid_data = train_data.split()

# Select only the most important 30000 words
MAX_VOCAB_SIZE = 30_000

# Build vocabularies

TEXT.build_vocab(train_data, 
                 max_size = MAX_VOCAB_SIZE, 
                 # Load pretrained embeddings
                 vectors = "glove.6B.100d", 
                 # Set unknown vectors
                 unk_init = torch.Tensor.normal_)

#TEXT.build_vocab(train_data)
LABEL.build_vocab(train_data)

BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create PyTorch iterators to use in training/evaluation/testing
train_iterator,valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data,valid_data, test_data), 
    batch_size = BATCH_SIZE, 
    device = device)

class CNN_Text(nn.Module):
  ''' Define network architecture and forward path. '''
  def __init__(self, vocab_size, 
                vector_size, n_filters, 
                filter_sizes, output_dim, 
                dropout, pad_idx):
      
      super().__init__()
      # Create word embeddings from the input words     
      self.embedding = nn.Embedding(vocab_size, vector_size, 
                                    padding_idx = pad_idx)
      
      
      # Specify convolutions with filters of different sizes (fs)
      self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                            out_channels = n_filters, 
                                            kernel_size = (fs, vector_size)) 
                                  for fs in filter_sizes])
      
      # Add a fully connected layer for final predicitons
      self.linear1 = nn.Linear(len(filter_sizes) * n_filters, output_dim)


      
      # Drop some of the nodes to increase robustness in training
      self.dropout = nn.Dropout(dropout)
   
      
      
  def forward(self, text):



      '''Forward path of the network.'''       
      # Get word embeddings and formt them for convolutions

      embedded1 = self.embedding(text)
      embedded = embedded1.unsqueeze(1)



      
      #model 1

      # Perform convolutions and apply activation functions
      conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

      #Pooling layer to reduce dimensionality    
      pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

      cat = self.dropout(torch.cat(pooled, dim = 1))

      cat = self.linear1(cat)
      #print(cat.shape)
      cnn1 =  torch.sigmoid(cat)
      #print("torch.cat(pooled, dim = 1)",torch.cat(conved, dim = 2).shape)

      return cnn1

class LSTM(nn.Module):
  ''' Define network architecture and forward path. '''
  def __init__(self, vocab_size, 
                vector_size, n_filters, 
                filter_sizes, output_dim, 
                dropout, pad_idx):
      
      super().__init__()
      # Create word embeddings from the input words     
      self.embedding = nn.Embedding(vocab_size, vector_size, 
                                    padding_idx = pad_idx)
      
      EMBEDDING_DIM = 100
      self.lstm = nn.LSTM(EMBEDDING_DIM, 64,num_layers=2,dropout=0.5,batch_first= True,bidirectional= True,bias = True)
      self.linear = nn.Linear(166400, 1)

      

  def init_hidden1(self, bsz):
    
    weight = next(self.parameters())
    
    return (weight.new_zeros(4, bsz, 64).to(device),weight.new_zeros(4, bsz, 64).to(device))
         
      
      
  def forward(self,text,hidden):



      '''Forward path of the network.'''       
      # Get word embeddings and formt them for convolutions

      embedded1 = self.embedding(text)


      #model 2

      #print("embedded bi",embedded1.shape)
      self.lstm.flatten_parameters()
      output, hidden  = self.lstm(embedded1, hidden)

      #print(output.shape)

      output = output.reshape(output.shape[0], output.shape[1]*output.shape[2])
      #print("output bi",output.shape)

      cat = self.linear(output)

      #cat = self.linear(cat)
      #print(cat.shape)
      lstm1 = torch.sigmoid(cat)

      return lstm1,hidden

class CNN_RNN(nn.Module):
  ''' Define network architecture and forward path. '''
  def __init__(self, vocab_size, 
                vector_size, n_filters, 
                filter_sizes, output_dim, 
                dropout, pad_idx):
      
      super().__init__()
      # Create word embeddings from the input words     
      self.embedding = nn.Embedding(vocab_size, vector_size, 
                                    padding_idx = pad_idx)
      
      
      # Specify convolutions with filters of different sizes (fs)
      self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, 
                                            out_channels = n_filters, 
                                            kernel_size = (fs, vector_size)) 
                                  for fs in filter_sizes])
      
     

      # Drop some of the nodes to increase robustness in training
      self.dropout = nn.Dropout(dropout).to(device)
      
      self.lstm = nn.LSTM(3891, 128,batch_first= True).to(device)
      self.linear = nn.Linear(12800, 1)


  def init_hidden(self, bsz):
    
    weight = next(self.parameters())
    
    return (weight.new_zeros(1, bsz, 128).to(device),weight.new_zeros(1, bsz, 128).to(device))


      
  def forward(self, text,hidden):
      '''Forward path of the network.'''       
      # Get word embeddings and formt them for convolutions

      embedded = self.embedding(text).unsqueeze(1)

      # Perform convolutions and apply activation functions
      conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

      cat = self.dropout(torch.cat(conved, dim = 2))
      #print("cat lstm",cat.shape)


      self.lstm.flatten_parameters()

      output, hidden  = self.lstm(cat, hidden)


      output = output.reshape(output.shape[0], output.shape[1]*output.shape[2])
      #print("output lstm",output.shape)

      cat = self.linear(output)


      return torch.sigmoid(cat), hidden

def accuracy(preds, y):
    """ Return accuracy per batch. """
    correct = (torch.round(preds) == y).float() 
    return correct.sum() / len(correct)

def epoch_time(start_time, end_time):
    '''Track training time. '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train2(model2, iterator1, optimizer2, criterion):
    '''Train the model with specified data, optimizer, and loss function. '''
    epoch_loss = 0
    epoch_acc = 0
    
    model2.train()
    hidden = model2.init_hidden(BATCH_SIZE)

    for data in iterator1:

          if(data.text.shape[0] == BATCH_SIZE):


              optimizer2.zero_grad()
              hidden = repackage_hidden(hidden)
              predictions ,hidden = model2(data.text,hidden)
              predictions = predictions.squeeze(1)
              loss = criterion(predictions, data.label)
            
              acc = accuracy(predictions, data.label)
            
              # Backprop
              loss.backward(retain_graph=True)
            
              # Optimize the weights
              optimizer2.step()
            
              # Record accuracy and loss
              epoch_loss += loss.item()
              epoch_acc += acc.item()

      
        
    return epoch_loss / len(iterator1), epoch_acc / len(iterator1),hidden

def train3(model3, iterator1, optimizer3, criterion):
    '''Train the model with specified data, optimizer, and loss function. '''
    epoch_loss = 0
    epoch_acc = 0
    
    model3.train()
    hidden = model3.init_hidden1(BATCH_SIZE)

    for data in iterator1:

          if(data.text.shape[0] == BATCH_SIZE):

        

              optimizer3.zero_grad()
              hidden = repackage_hidden(hidden)
              predictions ,hidden = model3(data.text,hidden)
              predictions = predictions.squeeze(1)
              loss = criterion(predictions, data.label)
            
              acc = accuracy(predictions, data.label)
            
              # Backprop
              loss.backward(retain_graph=True)
            
              # Optimize the weights
              optimizer3.step()
            
              # Record accuracy and loss
              epoch_loss += loss.item()
              epoch_acc += acc.item()

      
        
    return epoch_loss / len(iterator1), epoch_acc / len(iterator1),hidden



def train1(model, iterator, optimizer, criterion):
    '''Train the model with specified data, optimizer, and loss function. '''
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:

          if(batch.text.shape[0] == BATCH_SIZE):

        
              # Reset the gradient to not use them in multiple passes 
              optimizer.zero_grad()
              

              predictions = model(batch.text).squeeze(1)
              

              
              loss = criterion(predictions, batch.label)
              
              acc = accuracy(predictions, batch.label)
              
              # Backprop
              loss.backward(retain_graph=True)
              
              # Optimize the weights
              optimizer.step()
              
              # Record accuracy and loss
              epoch_loss += loss.item()
              epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model1,model2,model3,hidden1, hidden2,iterator1,alpha1,alpha2,alpha3,criterion):
    '''Evaluate model performance. '''
    epoch_loss = 0
    epoch_acc = 0
    
    # Turn off dropout while evaluating
    model1.eval()
    model2.eval()
    model3.eval()

    
    # No need to backprop in eval
    with torch.no_grad():
    
        for batch in iterator1:
          
          if(batch.text.shape[0] == BATCH_SIZE):
          

              predictions1 = model1(batch.text).squeeze(1)
              
              hidden1 = repackage_hidden(hidden1)
              predictions2 ,hidden1 = model2(batch.text,hidden1)
              predictions2 = predictions2.squeeze(1)

              hidden2 = repackage_hidden(hidden2)
              predictions3 ,hidden2 = model3(batch.text,hidden2)
              predictions3 = predictions3.squeeze(1)




              predictions = (alpha1*predictions1 + alpha2*predictions2 + alpha3*predictions3)/(alpha1 + alpha2 + alpha3)
              loss = criterion(predictions, batch.label)
              
              acc = accuracy(predictions, batch.label)

              epoch_loss += loss.item()
              epoch_acc += acc.item()
        
    return epoch_loss / len(iterator1), epoch_acc / len(iterator1)

# Vocabulary size
INPUT_DIM = len(TEXT.vocab)

# Vector size (lower-dimensional repr. of each word)
EMBEDDING_DIM = 100

# Number of filters
N_FILTERS = 100

# N-grams that we want to analuze using filters
FILTER_SIZES = [ 3, 4, 5]

# Output of the linear layer (prob of a negative review)
OUTPUT_DIM = 1

# Proportion of units to drop
DROPOUT = 0.5

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]



# Initialize model and load pre-trained embeddings
model1 = CNN_Text(INPUT_DIM, EMBEDDING_DIM, 
            N_FILTERS, FILTER_SIZES, 
            OUTPUT_DIM, DROPOUT, PAD_IDX)

model1.embedding.weight.data.copy_(TEXT.vocab.vectors)


model2 = CNN_RNN(INPUT_DIM, EMBEDDING_DIM, 
            N_FILTERS, FILTER_SIZES, 
            OUTPUT_DIM, DROPOUT, PAD_IDX)
'''
model2 = model3 = LSTM(INPUT_DIM, EMBEDDING_DIM, 
            N_FILTERS, FILTER_SIZES, 
            OUTPUT_DIM, DROPOUT, PAD_IDX)
'''
model2.embedding.weight.data.copy_(TEXT.vocab.vectors)

model3 = LSTM(INPUT_DIM, EMBEDDING_DIM, 
            N_FILTERS, FILTER_SIZES, 
            OUTPUT_DIM, DROPOUT, PAD_IDX)

model3.embedding.weight.data.copy_(TEXT.vocab.vectors)

# Zero the initial weights of the UNKnown and padding tokens.
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

# The string token used as padding. Default: “<pad>”.
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model1.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model1.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model1 = model1.to(device)


model2.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model2.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model2 = model2.to(device)


model3.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model3.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
model3 = model3.to(device)

# Network optimizer
optimizer1 = optim.Adam(model1.parameters())
optimizer2 = optim.Adam(model2.parameters())
optimizer3 = optim.Adam(model3.parameters())



# Loss function
criterion = nn.BCELoss()

model1 = model1.to(device)
model2 = model2.to(device)
model3 = model3.to(device)


criterion = criterion.to(device)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# Training loop
N_EPOCHS = 10

best_valid_loss = float('inf')
val_loss = []
val_acc = []

tr_loss1 = []
tr_acc1 = []

tr_loss2 = []
tr_acc2 = []

tr_loss3 = []
tr_acc3 = []
best_epoch = 0

for epoch in range(N_EPOCHS):
    
    # Calculate training time
    start_time = time.time()
    
    # Get epoch losses and accuracies 

    train_loss1, train_acc1 = train1(model1, train_iterator, optimizer1, criterion)
    train_loss2, train_acc2,hidden1 = train2(model2, train_iterator, optimizer2, criterion)
    train_loss3, train_acc3,hidden2 = train3(model3, train_iterator, optimizer3, criterion)

    alpha1 = (1-train_loss1)/(3 -train_loss1-train_loss2-train_loss3 )
    alpha2 = (1-train_loss2)/(3 -train_loss1-train_loss2-train_loss3 )
    alpha3 = (1-train_loss3)/(3 -train_loss1-train_loss2-train_loss3 )


    valid_loss, valid_acc = evaluate(model1,model2,model3,hidden1,hidden2,valid_iterator,alpha1,alpha2,alpha3, criterion)
    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # Save training metrics
    val_loss.append(valid_loss)
    val_acc.append(valid_acc)

    tr_loss1.append(train_loss1)
    tr_acc1.append(train_acc1)

    tr_loss2.append(train_loss2)
    tr_acc2.append(train_acc2)

    tr_loss3.append(train_loss3)
    tr_acc3.append(train_acc3)
    
    if valid_loss < best_valid_loss:
        best_epoch = epoch
        best_valid_loss = valid_loss
        #torch.save(model1.state_dict(), 'CNN-ensemble2-IMDB-model.pt')
        #torch.save(model2.state_dict(), 'LSTM-ensemble2-IMDB-model.pt')
        #torch.save(model3.state_dict(), 'CNN-RNN-ensemble2-IMDB-model.pt')
    
    print(f'Epoch: {epoch+1:2} | Epoch Time: {epoch_mins}m {epoch_secs}s')

    print(f'\tTrain Loss1: {train_loss1:.3f} | Train Acc1: {train_acc1*100:.2f}%')
    print(f'\tTrain Loss2: {train_loss2:.3f} | Train Acc2: {train_acc2*100:.2f}%')
    print(f'\tTrain Loss3: {train_loss3:.3f} | Train Acc3: {train_acc3*100:.2f}%')


    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

print("best_valid_loss", best_valid_loss)
print("best_epoch", best_epoch)

# Plot accuracy and loss
plt.figure(1)
plt.plot(val_loss, label='Validation loss')
plt.plot(tr_loss1, label='Training loss for CNN')
plt.plot(tr_loss2, label='Training loss for CNN-RNN')
plt.plot(tr_loss3, label='Training loss for LSTM')
plt.title('Loss for Ensemble2')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_ensemble2_SST')
plt.close
plt.figure(2)
plt.plot(val_acc, label='Validation accuracy')
plt.plot(tr_acc1, label='Training accuracy for CNN')
plt.plot(tr_acc2, label='Training accuracy for CNN-RNN')
plt.plot(tr_acc3, label='Training accuracy for LSTM')
plt.title('Accuracy for Ensemble2')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('Acc_ensemble2_SST')
plt.close

# Training loop
N_EPOCHS = best_epoch

best_valid_loss = float('inf')


tr_loss1 = []
tr_acc1 = []

tr_loss2 = []
tr_acc2 = []

tr_loss3 = []
tr_acc3 = []

for epoch in range(N_EPOCHS):
    
    # Calculate training time
    start_time = time.time()
    
    train_loss1, train_acc1 = train1(model1, train_iterator, optimizer1, criterion)
    train_loss2, train_acc2,hidden1 = train2(model2, train_iterator, optimizer2, criterion)
    train_loss3, train_acc3,hidden2 = train3(model3, train_iterator, optimizer3, criterion)

    alpha1 = (1-train_loss1)/(3 -train_loss1-train_loss2-train_loss3 )
    alpha2 = (1-train_loss2)/(3 -train_loss1-train_loss2-train_loss3 )
    alpha3 = (1-train_loss3)/(3 -train_loss1-train_loss2-train_loss3 )

    
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    # Save training metrics
    
    tr_loss1.append(train_loss1)
    tr_acc1.append(train_acc1)

    tr_loss2.append(train_loss2)
    tr_acc2.append(train_acc2)

    tr_loss3.append(train_loss3)
    tr_acc3.append(train_acc3)
    
   


    print(f'\tTrain Loss1: {train_loss1:.3f} | Train Acc1: {train_acc1*100:.2f}%')
    print(f'\tTrain Loss2: {train_loss2:.3f} | Train Acc2: {train_acc2*100:.2f}%')
    print(f'\tTrain Loss3: {train_loss3:.3f} | Train Acc3: {train_acc3*100:.2f}%')

    
test_loss, test_acc = evaluate(model1,model2,model3,hidden1,hidden2,test_iterator,alpha1,alpha2,alpha3, criterion)
print(f'\tTest_loss: {test_loss:.3f} | Test_acc: {test_acc*100:.2f}%')


