local Sequential2, _ = torch.class('Sequential2', 'nn.Sequential')

function Sequential2:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if torch.typename(currentModule) == 'ErrorFeedback' then
         currentGradOutput = gradOutput
      end
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function Sequential2:backward(input, gradOutput, scale, backprop)
   scale = scale or 1
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if torch.typename(currentModule) == 'ErrorFeedback' then
        if not backprop then
          currentGradOutput = currentModule:backward(previousModule.output, gradOutput, scale)
          currentModule.gradInput = currentGradOutput
        end
      else
        if torch.typename(currentModule) == 'SpatialConvolution' or torch.typename(currentModule) == 'Linear' then
          currentModule.backprop = backprop
        end
        currentGradOutput = currentModule:backward(previousModule.output, currentGradOutput, scale)
        currentModule.gradInput = currentGradOutput
      end
          
      currentModule = previousModule
   end
   currentGradOutput = currentModule:backward(input, currentGradOutput, scale)
   self.gradInput = currentGradOutput
      
   return currentGradOutput
end

function Sequential2:accUpdateGradParameters(input, gradOutput, lr)

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if torch.typename(currentModule) == 'ErrorFeedback' then
         currentGradOutput = gradOutput
      end
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end

   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end


function Sequential2:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'Sequential2'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
