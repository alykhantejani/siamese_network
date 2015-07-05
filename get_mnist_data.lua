require 'paths'
require 'torch'

mnist = {}

function mnist.download(dataset)
    dataset = {}
    dataset.remote_path = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
    dataset.root_folder = 'mnist.t7'
    dataset.trainset_path = paths.concat(mnist.path_dataset, 'train_32x32.t7')
    dataset.testset_path = paths.concat(mnist.path_dataset, 'test_32x32.t7')

   if not paths.filep(dataset.trainset_path) or not paths.filep(dataset.testset_path) then
      local tarfile = paths.basename(dataset.remote_path)
      os.execute('wget ' .. dataset.remote_path .. '; ' .. 'tar xvf ' .. tar .. '; rm ' .. tar)
   end
end

function mnist.load_normalized_dataset(filename)
    local file = torch.load(filename, 'ascii')
    
    local dataset = {}
    dataset.data = file.data:type(torch.getdefaulttensortype())
    dataset.labels = file.labels

    local std = std_ or dataset.data:std()
    local mean = mean_ or dataset.data:mean()
    dataset.data:add(-mean)
    dataset.data:mul(1.0/std)
    return mean, std
    
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
end

function mnist.load_siamese_dataset(filename)
    local file = torch.load(filename, 'ascii')
    
    data = file.data:type(torch.getdefaulttensortype())
    labels = file.labels

    local std = std_ or data:std()
    local mean = mean_ or data:mean()
    data:add(-mean)
    data:mul(1.0/std)
    return mean, std
    
  -- now we make the pairs

  data = dataset.data
  labels = dataset.labels
  shuffle = torch.randperm(data:size(1))
  max_index = data:size(1)
  if max_index % 2 ~= 0 then
    max_index = max_index - 1
  end


  paired_data = torch.Tensor(max_index/2, 2, data:size(2), data:size(3), data:size(4))
  paired_data_labels = torch.Tensor(max_index/2)
  index = 1

  for i = 1,max_index,2 do
    paired_data[index][i] = data[shuffle[i]]:clone()
    paired_data[index][i + 1] = data[shuffle[i + 1]]:clone()
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

end