require 'paths'
require 'torch'

mnist = {}
mnist.remote_path = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
mnist.root_folder = 'data/mnist.t7'
mnist.trainset_path = paths.concat(mnist.root_folder, 'train_32x32.t7')
mnist.testset_path = paths.concat(mnist.root_folder, 'test_32x32.t7')


function mnist.download(dataset)
   if not paths.filep(mnist.trainset_path) or not paths.filep(mnist.testset_path) then
      local tarfile = paths.basename(mnist.remote_path)
      os.execute('wget ' .. mnist.remote_path .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.load_normalized_dataset(filename, mean_, std_)
    local file = torch.load(filename, 'ascii')
    
    local dataset = {}
    dataset.data = file.data:type(torch.getdefaulttensortype())
    dataset.labels = file.labels

    local std = std_ or dataset.data:std()
    local mean = mean_ or dataset.data:mean()
    dataset.data:add(-mean);
    dataset.data:mul(1.0/std);

    dataset.std = std
    dataset.mean = mean
    
    function dataset:size()
      return dataset.data:size(1)
    end

    local class_count = 0
    local classes = {}
    for i=1, dataset.labels:size(1) do
      if classes[dataset.labels[i]] == nil then
        class_count = class_count + 1
        table.insert(classes, dataset.labels[i])
      end
    end

    dataset.class_count = class_count
    
    --The dataset has to be indexable by the [] operator so this next bit handles that
    setmetatable(dataset, {__index = function(self, index)
                          local input = self.data[index]
                          local class = self.labels[index]
                          local label_vector = torch.zeros(self.class_count)
                          label_vector[class] = 1
                          local example = {input, label_vector}
                          return example
                          end })
    return dataset
end

function mnist.load_siamese_dataset(filename, mean_, std_)
  local file = torch.load(filename, 'ascii')
  
  data = file.data:type(torch.getdefaulttensortype())
  labels = file.labels

  local std = std_ or data:std()
  local mean = mean_ or data:mean()
  data:add(-mean);
  data:mul(1.0/std);
    
  shuffle = torch.randperm(data:size(1))
  max_index = data:size(1)
  if max_index % 2 ~= 0 then
    max_index = max_index - 1
  end

  -- now we make the pairs (tensor of size (30000,2,1,32,32) for training data)
  paired_data = torch.Tensor(max_index/2, 2, data:size(2), data:size(3), data:size(4))
  paired_data_labels = torch.Tensor(max_index/2)
  index = 1

  for i = 1,max_index,2 do
    paired_data[index][1] = data[shuffle[i]]:clone()
    paired_data[index][2] = data[shuffle[i + 1]]:clone()
    if labels[shuffle[i]] == labels[shuffle[i+1]] then
      paired_data_labels[index] = 1
    else
      paired_data_labels[index] = -1
    end
    index = index + 1
  end 

  local dataset = {}
  dataset.data = paired_data
  dataset.labels = paired_data_labels
  dataset.std = std
  dataset.mean = mean

  function dataset:size()
    return dataset.data:size(1)
  end

  local class_count = 0
  local classes = {}
  for i=1, dataset.labels:size(1) do
    if classes[dataset.labels[i]] == nil then
      class_count = class_count + 1
      table.insert(classes, dataset.labels[i])
    end
  end

  dataset.class_count = class_count
  
  --The dataset has to be indexable by the [] operator so this next bit handles that
  setmetatable(dataset, {__index = function(self, index)
                        local input = self.data[index]
                        local label = self.labels[index]
                        local example = {input, label}
                        return example
                        end })

  return dataset
end