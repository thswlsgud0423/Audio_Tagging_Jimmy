import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, input_channels=1, img_height=128, img_width=216, num_classes=50,
                 map_to_seq_hidden=128, rnn_hidden_size=216, num_rnn_layers=3, dropout=0.3):
        super(CRNN, self).__init__()
        
        # CNN backbone
        self.cnn, (output_channels, output_height, output_width) = self._cnn_backbone(
            input_channels, img_height, img_width
        )
        
        # Map CNN output to sequence
        self.map_to_seq = nn.Linear(output_channels * output_height, map_to_seq_hidden)
        
        # Recurrent layers
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
        
        cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Halves height and width
            nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # Halves height and width
            nn.Dropout(0.3),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halves height, keeps width
            nn.Dropout(0.6),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # Halves height, keeps width
            nn.Dropout(0.5)
        )

        # Calculate output shape
        final_height = img_height // (2 * 2 * 2 * 2)  # Height halved 4 times
        final_width = img_width // (2 * 2)  # Width halved 2 times
        output_shape = (512, final_height, final_width)

        return cnn, output_shape


    def forward(self, x):
        
        x = self.cnn(x)
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels * height, width).permute(2, 0, 1)
        x = self.map_to_seq(x)
        x, _ = self.rnn1(x)
        x = self.fc(x[-1])
        return x