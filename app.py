class CNN_LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len=10):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # Transpose for CNN (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])
