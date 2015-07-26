require 'nn'
require 'dataset'
require 'gnuplot'
require 'image'

-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-gpu", false, "use gpu")
cmd:option("-test_data", "", "test data file (.t7)")
cmd:option("-dataset_attributes", "", "dataset attributes (mean & std)")
cmd:option("-pretrained_model", "", "pretained model (.net)")
cmd:option("-criterion", "", "criterion file")
cmd:option("-out", "out.png", "out file (.png)")
cmd:option("-log", "out.log", "output log file")

params = cmd:parse(arg)

if log ~= "" then
    cmd:log(params.log, params)
end
-----------------------------------------------------------------------------
-------------------------- Load test data -----------------------------------
-----------------------------------------------------------------------------
cmd:text("")
dataset_attributes = torch.load(params.dataset_attributes)
test_data = mnist.load_normalized_dataset(params.test_data, dataset_attributes.mean, dataset_attributes.std)
model = torch.load(params.pretrained_model)
criterion = torch.load(params.criterion)

deploy_model = model.modules[2].modules[1] --get one branch from the parallel table

x = torch.Tensor(test_data:size())
y = torch.Tensor(test_data:size())

for sample = 1,test_data:size(),1 do
    img = test_data[sample][1]
    label = test_data[sample][2]
    embedding = deploy_model:forward(img)

    x[sample] = embedding[1]
    y[sample] = embedding[2]
end

gnuplot.pngfigure(params.out)
gnuplot.plot(x,y,'.')
gnuplot.plotflush()
