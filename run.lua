require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'dataset'


-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-gpu", false, "use gpu")
cmd:option("-training_data", mnist.trainset_path, "training data (.t7) file")
cmd:option("-log", "out.log", "output log file")
cmd:option("-batch_size", 50, "batch size")
cmd:option("-learning_rate", 0.001, "learning_rate")
cmd:option("-snapshot_dir", "", "snapshot directory")

params = cmd:parse(arg)

logger = optim.Logger(params.log)
training_data = mnist.load_normalized_dataset(params.training_data)

epoch = 0
batch_size = params.batch_size
learning_rate = params.learning_rate


--Load model and criterion

-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)
parameters, grad_parameters = model:getParameters()

function train_one_epoch(dataset)
      
    local time = sys.clock()
    --train one epoch of the dataset
    print("training epoch #" .. epoch)
   
    for mini_batch_start = 1, dataset:size(), batch_size do --for each mini-batch
        xlua.progress(mini_batch, dataset:size()) --display progress

        local inputs = {}
        local labels = {}
        --create a mini_batch
        for i = mini_batch_start, math.min(mini_batch_start + batch_size - 1, dataset:size()) do 
            local input = dataset[i][1]:clone() -- the tensor containing two images 
            local label = dataset[i][2]:clone() -- +/- 1
            table.insert(inputs, input)
            table.insert(labels, label)
        end

        --create a closure to evaluate df/dX where x are the model parameters at a given point
        --and df/dx is the gradient of the loss wrt to thes parameters

        local func_eval = 
        function(x)

                --update the model parameters (copy x in to parameters)
                if x ~= parameters then
                    parameters:copy(x) 
                end

                grad_parameters:zero() --reset gradients

                local avg_error = 0 -- the average error of all criterion outs

                --evaluate for complete mini_batch
                for i = 1, #inputs do
                    local output = model:forward(inputs[i])
                    local err = criterion:forward(output, labels[i])
                    avg_error = avg_error + err

                    --estimate dLoss/dW
                    local dloss_dout = criterion:backward(output, labels[i])
                    model:backward(inputs[i], dloss_dout)
                end

                grad_parameters:div(#inputs)
                avg_error = avg_error / #inputs

                return avg_error, grad_parameters
        end


        config = {learningRate = opt.learningRate,
                  weightDecay = opt.weightDecay,
                  momentum = opt.momentum,
                  learningRateDecay = 5e-7}

        --This function updates the global parameters variable (which is a view on the models parameters)
        optim.sgd(func_eval, parameters, config)

    end

    -- time taken
    time = sys.clock() - time
    time = time / dataset:size()
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms')

    epoch = epoch + 1
end