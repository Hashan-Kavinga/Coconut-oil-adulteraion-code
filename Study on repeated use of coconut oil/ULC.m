function ULC(digit1, index)
    digit2 = index + 10^numel(digit1);
    digit2 = int2str(digit2);
    digit2 = digit2(2:numel(digit2));
    for b = 1 : numel(digit2)+4
        fprintf('\b');
    end
    fprintf('%s/100',digit2);
end