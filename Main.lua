require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
local c = require 'trepl.colorize'

dofile './Models/BCECriterion.lua'

----------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:addTime()
cmd:text()
cmd:text('Training a convolutional network for visual classification')
cmd:text()
cmd:text('==>Options')

cmd:text('===>Model options')
cmd:option('-modelsFolder',       './Models/',            'Models Folder')
cmd:option('-network',            'mlp.lua',              'Model file - must return valid network.')
cmd:option('-criterion',          'bce',                  'criterion, ce(cross-entropy) or bce(binary cross-entropy)')
cmd:option('-eps',                0,                      'adversarial regularization magnitude (fast-sign-method a.la Goodfellow)')
cmd:option('-dropout',            0,                      'apply dropout')
cmd:option('-batchnorm',          0,                      'apply batch normalization')
cmd:option('-nonlin',             'tanh',                 'nonlinearity, (tanh,sigm,relu)')
cmd:option('-num_layers',         2,                      'number of hidden layers (if applicable)')
cmd:option('-num_hidden',         800,                    'number of hidden neurons (if applicable)')
cmd:option('-bias',               1,                      'use bias or not')
cmd:option('-rfb_mag',            0,                      'random feedback magnitude, 0=auto scale')

cmd:text('===>Training Regime')
cmd:option('-LR',                 0.0001,                 'learning rate')
cmd:option('-LRDecay',            0,                      'learning rate decay (in # samples)')
cmd:option('-weightDecay',        0.0,                    'L2 penalty on the weights')
cmd:option('-momentum',           0.0,                    'momentum')
cmd:option('-batchSize',          64,                     'batch size')
cmd:option('-optimization',       'rmsprop',              'optimization method')
cmd:option('-epoch',              300,                    'number of epochs to train, -1 for unbounded')
cmd:option('-epoch_step',         -1,                     'learning rate step, -1 for no step, 0 for auto, >0 for multiple of epochs to decrease')
cmd:option('-gradient',           'dfa',                  'gradient for learning (bp, fa or dfa)')
cmd:option('-maxInNorm',           400,                   'max norm on incoming weights')
cmd:option('-maxOutNorm',          400,                   'max norm on outgoing weights')
cmd:option('-accGradient',         0,                     'accumulate back-prop and adversarial gradient (eps>0)')

cmd:text('===>Platform Optimization')
cmd:option('-threads',            8,                      'number of threads')
cmd:option('-type',               'cuda',                 'float or cuda')
cmd:option('-devid',              1,                      'device ID (if using CUDA)')
cmd:option('-nGPU',               1,                      'num of gpu devices used')
cmd:option('-constBatchSize',     false,                  'do not allow varying batch sizes - e.g for ccn2 kernel')

cmd:text('===>Save/Load Options')
cmd:option('-load',               '',                     'load existing net weights')
cmd:option('-save',               os.date():gsub(' ',''), 'save directory')

cmd:text('===>Data Options')
cmd:option('-dataset',            'MNIST',                'Dataset - Cifar10, Cifar100, STL10, SVHN, MNIST')
cmd:option('-normalization',      'scale',                'scale - between 0 and 1, simple - whole sample, channel - by image channel, image - mean and std images')
cmd:option('-format',             'rgb',                  'rgb or yuv')
cmd:option('-whiten',             false,                  'whiten data')
cmd:option('-augment',            false,                  'Augment training data')
cmd:option('-preProcDir',         './PreProcData/',       'Data for pre-processing (means,P,invP)')
cmd:option('-validate',           false,                  'use validation set for testing instead of test set')
cmd:option('-datapath',           './Datasets/',          'data set directory'

cmd:text('===>Misc')
cmd:option('-visualize',          0,                      'visualizing results')

opt = cmd:parse(arg or {})
opt.network = opt.modelsFolder .. paths.basename(opt.network, '.lua')
opt.preProcDir = paths.concat(opt.preProcDir, opt.dataset .. '/')
os.execute('mkdir -p ' .. opt.preProcDir)
torch.setnumthreads(opt.threads)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.augment then
    require 'image'
end
----------------------------------------------------------------------

-- classes
local data = require 'Data'
local classes = data.Classes

ninput = data.TrainData.Data:nElement() / data.TrainData.Data:size(1)
noutput = #classes

-- Model + Loss:
local model = require(opt.network)
local loss
if opt.criterion == 'bce' then
  loss = BCECriterion()
else
  loss = nn.ClassNLLCriterion()
end

----------------------------------------------------------------------

-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

local AllowVarBatch = not opt.constBatchSize


----------------------------------------------------------------------


-- Output files configuration
os.execute('mkdir -p ' .. opt.save)
cmd:log(opt.save .. '/Log.txt', opt)
local netFilename = paths.concat(opt.save, 'Net')
local logFilename = paths.concat(opt.save,'ErrorRate.log')
local optStateFilename = paths.concat(opt.save,'optState')
local Log = optim.Logger(logFilename)
----------------------------------------------------------------------

local TensorType = 'torch.FloatTensor'
if opt.type =='cuda' then
    require 'cutorch'
    cutorch.setDevice(opt.devid)
    model:cuda()
    loss = loss:cuda()
    TensorType = 'torch.CudaTensor'
end


---Support for multiple GPUs - currently data parallel scheme
if opt.nGPU > 1 then
    local net = model
    model = nn.DataParallelTable(1)
    for i = 1, opt.nGPU do
        cutorch.setDevice(i)
        model:add(net:clone():cuda(), i)  -- Use the ith GPU
    end
    cutorch.setDevice(opt.devid)
end

-- Optimization configuration
local Weights,Gradients = model:getParameters()

local savedModel --savedModel - lower footprint model to save
if opt.nGPU > 1 then
    model:syncParameters()
    savedModel = model.modules[1]:clone('weight','bias','running_mean','running_std')
else
    savedModel = model:clone('weight','bias','running_mean','running_std')
end

----------------------------------------------------------------------
print '==> Network'
print(model)
print('==>' .. Weights:nElement() ..  ' Parameters')

print '==> Loss'
print(loss)


------------------Optimization Configuration--------------------------
local optimState = {
    learningRate = opt.LR,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.LRDecay
}
----------------------------------------------------------------------

local function SampleImages(images,labels)
    if not opt.augment then
        return images,labels
    else

        local sampled_imgs = images:clone()
        for i=1,images:size(1) do
            local sz = math.random(9) - 1
            local hflip = math.random(2)==1

            local startx = math.random(sz)
            local starty = math.random(sz)
            local img = images[i]:narrow(2,starty,32-sz):narrow(3,startx,32-sz)
            if hflip then
                img = image.hflip(img)
            end
            img = image.scale(img,32,32)
            sampled_imgs[i]:copy(img)
        end
        return sampled_imgs,labels
    end
end


------------------------------
local function Forward(Data, train, savestate)


  local MiniBatch = DataProvider.Container{
    Name = 'GPU_Batch',
    MaxNumItems = opt.batchSize,
    Source = Data,
    ExtractFunction = SampleImages,
    TensorType = TensorType
  }

  local yt = MiniBatch.Labels
  local x = MiniBatch.Data

  local SizeData = Data:size()
  if not AllowVarBatch then SizeData = math.floor(SizeData/opt.batchSize)*opt.batchSize end

  local NumSamples = 0
  local NumBatches = 0
  local lossVal = 0

  while NumSamples < SizeData do
    MiniBatch:getNextBatch()
    local y, currLoss
    NumSamples = NumSamples + x:size(1)
    NumBatches = NumBatches + 1
    if opt.nGPU > 1 then
      model:syncParameters()
    end

    y = model:forward(x)
    
    currLoss = loss:forward(y,yt)
    if train then
      function feval()
        if opt.maxOutNorm < 400 or opt.maxInNorm < 400 then
          model:maxParamNorm(opt.maxOutNorm, opt.maxInNorm) -- affects params
        end
        model:zeroGradParameters()
        local dE_dy = loss:backward(y, yt)
        local dE_dx = model:backward(x, dE_dy, 1, opt.eps > 0)
        
        -- Train on adversarial samples
        if opt.eps > 0 then
          if opt.accGradient == 0 then
            model:zeroGradParameters()
          end
          x:add(torch.sign(dE_dx):mul(opt.eps)) -- Add perturbation
          y = model:forward(x)
          currLoss = loss:forward(y,yt)
          dE_dy = loss:backward(y, yt)
          model:backward(x, dE_dy, 1, false)
        end
        
        return currLoss, Gradients
      end
      _G.optim[opt.optimization](feval, Weights, optimState)
      
      if opt.bias == 0 then
        for i,layer in ipairs(model.modules) do
           if layer.bias then
              layer.bias:fill(0)
           end
        end
      end
    end

    lossVal = currLoss + lossVal

    if type(y) == 'table' then --table results - always take first prediction
      y = y[1]
    end

    confusion:batchAdd(y,yt)
    if train then
      xlua.progress(NumSamples, SizeData)
    end
    if math.fmod(NumBatches,100)==0 then
      collectgarbage()
    end
  end
  return(lossVal/math.ceil(SizeData/opt.batchSize))
end

------------------------------
local function Train(Data)
  model:training()
  return Forward(Data, true)
end

local function Test(Data)
  model:evaluate()
  return Forward(Data, false)
end
------------------------------

local train_err_mean = 1
local decay_count = 0
local finished = false
local epoch = 1
print '\n==> Starting Training\n'

while epoch ~= opt.epoch and finished == false do
    local tic = torch.tic()
    
    data.TrainData:shuffleItems()

    print(c.blue '==>'.." Epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ', learningRate = ' .. optimState.learningRate ..', momentum = ' .. opt.momentum ..', eps = ' .. opt.eps ..']')
    
    --Train
    confusion:zero()
    local LossTrain = Train(data.TrainData)
    torch.save(netFilename, savedModel)
    confusion:updateValids()
    local ErrTrain = (1-confusion.totalValid)
    
    train_err_mean = 0.75*train_err_mean + 0.25*ErrTrain
    if opt.epoch_step == 0 then
      if decay_count >= 8 or train_err_mean <= 0.0001 then
        finished = true
        print('Finished')
      elseif ErrTrain >= train_err_mean then
        optimState.learningRate = optimState.learningRate/2
        decay_count = decay_count + 1
        print('Decaying learning rate by a factor of 2')
      end
    elseif opt.epoch_step > 0 and (epoch % opt.epoch_step) == 0 then
      optimState.learningRate = optimState.learningRate/2
      decay_count = decay_count + 1
      print('Decaying learning rate by a factor of 2')
    end
  
    if opt.epoch_step < opt.epoch then
      if train_err_mean <= 0.0001 then
        finished = true
        print('Finished: zero training error reached')
      end
    end
        
    print(('Train error: '..c.cyan'%5.2f'..' %%,\t mean error: '..c.cyan'%5.2f'..' %%,\t loss: %.5f,\t time: %.2f s'):format(
        ErrTrain * 100, train_err_mean * 100, LossTrain, torch.toc(tic)))
    
    --Test
    confusion:zero()
    local LossTest = Test(data.TestData)
    confusion:updateValids()
    local ErrTest = (1-confusion.totalValid)

    print(('Test error:  '..c.cyan'%5.2f'..' %%,\t loss: %.5f'):format(
        ErrTest * 100, LossTest))
    
    Log:add{['Training Error']= ErrTrain, ['Test Error'] = ErrTest}
    if opt.visualize == 1 then
        Log:style{['Training Error'] = '-', ['Test Error'] = '-'}
        Log:plot()
    end

    epoch = epoch + 1
end
