function [index] =  RouletteWheelSelection(arrayInput)
% the bigger value the element, the higher probablity to be selected.
len = length(arrayInput);

% if input is one element then just return rightaway
if len ==1
    index =1;
    return;
end

if (~isempty(find(arrayInput<1, 1)))
    if (min(arrayInput) ~=0)
        [Y,I] = min(arrayInput);
    arrayInput = 1/Y*arrayInput;
    arrayInput(I) = round(arrayInput(I));  %
    else
    temp= arrayInput;
    temp(arrayInput==0) = inf;
    arrayInput = 1/min(temp)*arrayInput;
    end
end

temp = 0;
tempProb = zeros(1,len);

%Calculate cumulative probability
for i= 1:len
    tempProb(i) = temp + arrayInput(i);
    temp = tempProb(i);
end

% i = fix(rand*floor(tempProb(end)))+1.0;
i = fix(rand*floor(tempProb(end)))+0.9999999999999;
index = find(tempProb >= i, 1 );