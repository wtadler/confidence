function score = calcscoreconf(responses,section,conflevels)

n=numel(responses.tf(section,:));
%tf=responses.tf;
responses.tf(responses.tf==0) = -1;
score1 = sum(sum(responses.tf(section,:) .* responses.conf(section,:))); % with range n.trials * [-conflevels:conflevels]

score = 100 * (n * conflevels + score1)/(2 * n * conflevels);  % convert score1 into percentage