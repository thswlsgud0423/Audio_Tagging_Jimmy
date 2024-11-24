import torch.nn as nn



class ImprovedCRNN(nn.Module):
    def __init__(self, input_channels=1, img_height=128, img_width=128, num_classes=50,
                 map_to_seq_hidden=64, rnn_hidden_size=256, num_rnn_layers=3, dropout=0.2):
        super(ImprovedCRNN, self).__init__()
        
        # CNN backbone
        self.cnn, (output_channels, output_height, output_width) = self._cnn_backbone(
            input_channels, img_height, img_width
        )
        
        # Map CNN output to sequence
        self.map_to_seq = nn.Linear(output_channels * output_height, map_to_seq_hidden)
        
        # Recurrent layers with dropout
        self.rnn1 = nn.LSTM(
            map_to_seq_hidden,
            rnn_hidden_size,
            num_layers=num_rnn_layers,
            bidirectional=True,
            dropout=dropout,
            batch_first=False
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden_size * 2, num_classes)

    def _cnn_backbone(self, input_channels, img_height, img_width):
        """
        Builds the CNN backbone with proper output dimension adjustments.
        """
        cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Halves height and width
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Further halves height and width
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # Height divided by 2, width stays same
            nn.Dropout(0.4),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # Height divided by 2, width stays same
            nn.Dropout(0.4)
        )

        # Calculate output dimensions
        final_height = img_height // 16 - 1
        final_width = img_width // 4 - 1
        return cnn, (512, final_height, final_width)


    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels * height, width).permute(2, 0, 1)
        x = self.map_to_seq(x)
        x, _ = self.rnn1(x)
        x = self.fc(x[-1])  # Use only the last time step for classification
        return x