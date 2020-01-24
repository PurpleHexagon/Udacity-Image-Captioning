import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch_layer= nn.BatchNorm1d(embed_size, momentum = 0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return self.batch_layer(features)
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.embedded_layer = nn.Embedding(vocab_size, embed_size)
        self.linear_layer = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embedded_words = self.embedded_layer(captions[:, :-1])
        embedded_words = torch.cat((features.unsqueeze(1), embedded_words), 1)
        lstm_output, _ = self.lstm(embedded_words)
        
        return self.linear_layer(lstm_output)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for i in range(max_len):
            # the first value returned by LSTM is all of the hidden states throughout
            # the sequence. the second is just the most recent hidden state
            lstm_outputs, states = self.lstm(inputs, states)
            lstm_outputs = lstm_outputs.squeeze(1)
            linear_output = self.linear_layer(lstm_outputs)
            _, predicted_word = linear_output.max(1)
            sentence.append(predicted_word.item())
            
            # Prevent multiple end tokens which was happening sometimes
            if predicted_word == 1:
                break
            inputs = self.embedded_layer(predicted_word).unsqueeze(1)
        
        return sentence