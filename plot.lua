require 'nn'
require 'dataset'
require 'gnuplot'
require 'image'

function get_label(one_hot_vector)
    for i = 1,one_hot_vector:size()[1] do
        if one_hot_vector[i] == 1 then
            return i
        end
    end
    return 0
end
-----------------------------------------------------------------------------
--------------------- parse command line options ----------------------------
-----------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text("Argument")
cmd:argument("-test_data", "test data file (.t7)")
cmd:argument("-dataset_attributes", "dataset attributes (mean & std)")
cmd:argument("-pretrained_model", "pretained model (.net)")
cmd:text("Options")
cmd:option("-out", "out.png", "out file (.png)")
cmd:option("-gpu", false, "use gpu")
cmd:option("-log", "", "output log file")

params = cmd:parse(arg)

if params.log ~= "" then
    cmd:log(params.log, params)
end
-----------------------------------------------------------------------------
-------------------------- Load test data -----------------------------------
-----------------------------------------------------------------------------
cmd:text("")
dataset_attributes = torch.load(params.dataset_attributes)
test_data = mnist.load_normalized_dataset(params.test_data, dataset_attributes.mean, dataset_attributes.std)
model = torch.load(params.pretrained_model)

deploy_model = model.modules[2].modules[1] --get one branch from the parallel table

x = torch.Tensor(test_data:size())
y = torch.Tensor(test_data:size())

result_map = {}

for sample = 1, test_data:size() do
    label = get_label(test_data[sample][2])
    if result_map[label] == nil then
        result_map[label] = {}
    end
    table.insert(result_map[label], sample)
end

print("generating embeddings ")
plot_results = {}
for label,ids in pairs(result_map) do
    local x = torch.Tensor(#ids):fill(0)
    local y = torch.Tensor(#ids):fill(0)

    for id = 1,#ids do
        local img = test_data[ids[id]][1]
        local embedding = deploy_model:forward(img)
        x[id] = embedding[1]
        y[id] = embedding[2]
    end
    table.insert(plot_results, {tostring(label), x, y, '+'})
end
print("plotting...")
gnuplot.pngfigure(params.out)
gnuplot.grid()
gnuplot.xlabel('x')
gnuplot.ylabel('y')
gnuplot.axis({-2.25,2.25,-2.25,2.25})
gnuplot.plot(plot_results)
gnuplot.plotflush()
print("done.")