require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'

dofile './Models/Linear.lua'
dofile './Models/ErrorFeedback.lua'
dofile './Models/Sequential2.lua'

torch.include('dpnn', 'Module.lua')

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  --require 'cudnn'
  backend = cudnn
  cudnn.benchmark = true
  --cudnn.fastest = true
else
  backend = nn
end

local function AddNonLinearity(model)
  if opt.nonlin == "tanh" then
    model:add(backend.Tanh())
  elseif opt.nonlin == "sigm" then
    model:add(backend.Sigmoid())
  else
    model:add(backend.ReLU())
  end
end


local model = Sequential2()
local a = opt.rfb_mag

model:add(nn.View(ninput))

if opt.dropout >= 1 then
  model:add(nn.Dropout(0.1))
end
model:add(nn.Linear(ninput, opt.num_hidden))
if opt.batchnorm == 1 then
  model:add(nn.BatchNormalization(opt.num_hidden))
end
AddNonLinearity(model)

if opt.dropout == 1 then
  model:add(nn.Dropout(0.5))
end
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end

for i=1,opt.num_layers-1 do
  if opt.gradient == 'fa' then
    model:add(Linear(opt.num_hidden, opt.num_hidden, a))
  else
    model:add(nn.Linear(opt.num_hidden, opt.num_hidden))
  end
  if opt.batchnorm == 1 then
    model:add(nn.BatchNormalization(opt.num_hidden))
  end
  AddNonLinearity(model)
  if opt.dropout == 1 then
    model:add(nn.Dropout(0.5))
  end
  if opt.gradient == 'dfa' then
    model:add(ErrorFeedback(a))
  end
end
if opt.gradient == 'fa' then
  model:add(Linear(opt.num_hidden, noutput, a))
else
  model:add(nn.Linear(opt.num_hidden, noutput))
end

if opt.criterion == 'ce' then
  model:add(nn.LogSoftMax())
end

for k,v in pairs(model:findModules('nn.Linear')) do
  if opt.gradient == 'dfa' then
    if opt.nonlin == "tanh" or opt.nonlin == "sigm" then
      v.weight:zero()
      v.bias:zero()
    end
  end
end

for k,v in pairs(model:findModules('Linear')) do
  if opt.gradient == 'fa' then
    if opt.nonlin == "tanh" or opt.nonlin == "sigm" then
      v.weight:zero()
      v.bias:zero()
    end
  end
end

--print(#model:cuda():forward(torch.CudaTensor(1,3,32,32)))

return model
