require 'imge'
require 'nn'

--after a convolution the output width and height is
--owidth  = floor((width  + 2*padW - kW) / dW + 1)
--oheight = floor((height + 2*padH - kH) / dH + 1)

--Encoder/Embedding
--Input dims are 28x28          NOTE: change dims as inputs are 32x32
encoder = nn.Sequential()
encoder:add(nn.SpatialConvolution(1, 20, 5, 5)) -- 1 input image channel, 20 output channels, 5x5 convolution kernel (each feature map is 24x24)
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling with kernal 2x2 and a stride of 2 in each direction (feature maps are 12x12)
encoder:add(nn.SpatialConvolution(20, 50, 5, 5)) -- 20 input feature maps and output 50, 5x5 convolution kernel (feature maps are 8x8)
encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- max pooling (feature maps are 4x4)

encoder:add(nn.View(50*4*4)) --reshapes to view data at 50x4x4
encoder:add(nn.Linear(50*4*4, 500))
encoder:add(nn.ReLU())
encoder:add(nn.Linear(500, 10))
--encoding layer - go from 10 class out to 2-dimensional encoding
encoder:add(nn.Linear(10, 2))

--The siamese model
siamese_encoder = nn.ParallelTable()
siamese_encoder:add(encoder)
siamese_encoder:add(encoder:clone('weight','bias')) --clone the encoder and share the weight, bias



--The siamese model (inputs will be Tensors of shape (2, channel, height, width))
model = nn.Sequential()
model:add(nn.SplitTable(1)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
model:add(siamese_encoder)
model:add(nn.PairwiseDistance(2)) --L2 pariwise distance

margin = 1
criterion = nn.HingeEmbeddingCriterion(margin)
