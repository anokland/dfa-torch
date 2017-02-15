
require 'nn'
local THNN = require 'nn.THNN'

local BCECriterion, parent = torch.class('BCECriterion', 'nn.Criterion')

function BCECriterion:__init(sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end

    self.target = torch.zeros(1):long()
    self.sig = nn.Sigmoid()
    self.bce = nn.BCECriterion()
    self.buffer = torch.Tensor()
    self.input = torch.Tensor()
end

function BCECriterion:updateOutput(input, target)
   if type(target) == 'number' then
      if input:type() ~= 'torch.CudaTensor' then
         self.target = self.target:long()
      end
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   self.buffer:resizeAs(input):zero()
   self.input:resizeAs(input):zero()
   
   self.input = self.sig:forward(input)
   
   local indices = self.target:view(-1,1)
   self.buffer:scatter(2, indices, 1)
   
   self.bce:updateOutput(self.input, self.buffer)
   self.output = self.bce.output

   return self.output
end

function BCECriterion:updateGradInput(input, target)
   if type(target) == 'number' then
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   self.gradInput:resizeAs(input):zero()
   self.input:resizeAs(input):zero()
   
   self.input = self.sig:forward(input)

   local indices = self.target:view(-1,1)
   self.gradInput:scatter(2, indices, 1)
   self.gradInput:mul(-1)
   self.gradInput:add(self.input)
   
   return self.gradInput
end
