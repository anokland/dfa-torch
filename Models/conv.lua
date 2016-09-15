require 'nn'
require 'cunn'
require 'cudnn'
require 'dpnn'

dofile './Models/Linear.lua'
dofile './Models/SpatialConvolution.lua'
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

local nstates = {96,128,256}
local a = opt.rfb_mag

if opt.dropout == 1 then
  model:add(nn.Dropout(0.1))
end
if opt.gradient == 'fa' then
  model:add(SpatialConvolution(3, nstates[1], 5, 5, 1, 1, 2, 2, a))
else
  model:add(backend.SpatialConvolution(3, nstates[1], 5, 5, 1, 1, 2, 2))
end

if opt.batchnorm == 1 then
  model:add(backend.SpatialBatchNormalization(nstates[1],1e-3))
end
AddNonLinearity(model)
if opt.dropout == 1 or opt.dropout == 2 then
  model:add(nn.Dropout(0.25))
end

model:add(backend.SpatialMaxPooling(3,3,2,2))
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end
if opt.gradient == 'fa' then
  model:add(SpatialConvolution(nstates[1], nstates[2], 5, 5, 1, 1, 2, 2, a))
else
  model:add(backend.SpatialConvolution(nstates[1], nstates[2], 5, 5, 1, 1, 2, 2))
end

if opt.batchnorm == 1 or opt.batchnorm == 2 then
  model:add(backend.SpatialBatchNormalization(nstates[2],1e-3))
end
AddNonLinearity(model)
if opt.dropout == 1 or opt.dropout == 2 then
  model:add(nn.Dropout(0.25))
end

model:add(backend.SpatialMaxPooling(3,3,2,2))
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end
if opt.gradient == 'fa' then
  model:add(SpatialConvolution(nstates[2], nstates[3], 5, 5, 1, 1, 2, 2, a))
else
  model:add(backend.SpatialConvolution(nstates[2], nstates[3], 5, 5, 1, 1, 2, 2))
end
if opt.batchnorm == 1 then
  model:add(backend.SpatialBatchNormalization(nstates[3],1e-3))
end

AddNonLinearity(model)

if opt.dropout == 1 then
  model:add(nn.Dropout(0.5))
end

model:add(backend.SpatialMaxPooling(3,3,2,2))
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end
model:add(nn.View(nstates[3]*3*3))

if opt.gradient == 'fa' then
  model:add(Linear(nstates[3]*3*3, 2048, a))
else
  model:add(nn.Linear(nstates[3]*3*3, 2048))
end
if opt.batchnorm == 1 then
  model:add(nn.BatchNormalization(2048))
end
AddNonLinearity(model)

if opt.dropout == 1 then
  model:add(nn.Dropout(0.5))
end
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end
if opt.gradient == 'fa' then
  model:add(Linear(2048, 2048, a))
else
  model:add(nn.Linear(2048, 2048))
end
if opt.batchnorm == 1 then
  model:add(nn.BatchNormalization(2048))
end
AddNonLinearity(model)
if opt.dropout == 1 then
  model:add(nn.Dropout(0.5))
end
if opt.gradient == 'dfa' then
  model:add(ErrorFeedback(a))
end
if opt.gradient == 'fa' then
  model:add(Linear(2048, noutput, a))
else
  model:add(nn.Linear(2048, noutput))
end
if opt.criterion == 'ce' then
  model:add(nn.LogSoftMax())
end


for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  if opt.gradient == 'dfa' then
    if opt.nonlin == "tanh" or opt.nonlin == "sigm" then
      v.weight:zero()
      v.bias:zero()
    end
  end
end

for k,v in pairs(model:findModules('SpatialConvolution')) do
  if opt.gradient == 'fa' then
    if opt.nonlin == "tanh" or opt.nonlin == "sigm" then
      v.weight:zero()
      v.bias:zero()
    end
  end
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
