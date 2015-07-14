require 'torch';
require 'nn';
require 'optim';
require 'image';
require 'dataset';
require 'model';


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
cmd:option("-momentum", 0.9, "momentum")
cmd:option("-max_epochs", 2, "maximum epochs")
cmd:option("-snapshot_dir", "", "snapshot directory")
cmd:option("-snapshot", nil, "snapshot after how many iterations?")

params = cmd:parse(arg)

epoch = 0
batch_size = params.batch_size
--Load model and criterion

-- retrieve a view (same memory) of the parameters and gradients of these (wrt loss) of the model (Global)
parameters, grad_parameters = model:getParameters();

function train(data)
    local saved_criterion = false;
    for i = 1, params.max_epochs do
        --add random shuffling here
        train_one_epoch(data)

        if params.snapshot ~= nil && epoch % params.snapshot == 0 then -- epoch is global (gotta love lua :p)
            local filename = paths.concat(params.save, "_epoch_" .. epoch .. ".net")
            os.execute('mkdir -p ' .. sys.dirname(filename))
            torch.save(filename, model)        
            --must save std, mean and criterion?
            if not saved_criterion then
                local criterion_filename = paths.concat(params.save, "_criterion.net")
                torch.save(criterion_filename, criterion)
            end
        end
    end
end

function train_one_epoch(dataset)

    local time = sys.clock()
    --train one epoch of the dataset

    for mini_batch_start = 1, dataset:size(), batch_size do --for each mini-batch
        
        local inputs = {}
        local labels = {}
        --create a mini_batch
        for i = mini_batch_start, math.min(mini_batch_start + batch_size - 1, dataset:size()) do 
            local input = dataset[i][1]:clone() -- the tensor containing two images 
            local label = dataset[i][2] -- +/- 1
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

                grad_parameters:div(#inputs);
                avg_error = avg_error / #inputs;

                return avg_error, grad_parameters
        end


        config = {learningRate = params.learning_rate, momentum = params.momentum}

        --This function updates the global parameters variable (which is a view on the models parameters)
        optim.sgd(func_eval, parameters, config)
        
        xlua.progress(mini_batch_start, dataset:size()) --display progress
    end

    -- time taken
    time = sys.clock() - time
    print("time taken for 1 epoch = " .. (time * 1000) .. "ms, time taken to learn 1 sample = " .. ((time/dataset:size())*1000) .. 'ms')
    epoch = epoch + 1
end



print("loading dataset...")
mnist_dataset = mnist.load_siamese_dataset("/Users/aly/workspace/torch_sandbox/siamese_network/data/mnist.t7/train_32x32.t7")
print("dataset loaded")
train(mnist_dataset)