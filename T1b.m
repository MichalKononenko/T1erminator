clc;
clear all;

%bayesian learning stuffs brother
expct=@(x,px) sum(x.*px); 
N=30; %number of SQM iterations
%This par;t is the T1 simulation
TrueT1=2.6; %actual value of T1 that we will try to find using BL
noiseSTD=0.1; %STD of noise added to the model 
Pn=@(t) ((-2*(exp(-t/TrueT1)))+1)+noiseSTD*randn(1); %noisy model 
P=@(t,T) ((-2*(exp(-t/T)))+1); %noiseless model 
Tone=@(t,q) t./(log(2./(1-q))); %T1 based on time and polarization

%defining likelihood function to be a normal distribution
c=linspace(-1,1,1000)'; %possible polarization values
T1Space=@(t) Tone(t,c); %space transformed from polarization to T1 (not sure if this is needed) 
Likelihood=@(t,T) normpdf(c,P(t,T),noiseSTD); %t is experimental parameter (tau) 

%prior=@(tau, mean, std, T, D) ;

%Sequantial monte carlo

Ptest=zeros(N,1); %vector of measured polarizations
Ttest=zeros(N,1); %vector of expectation of T1
ttest=zeros(N,1); %vector of experimental parameters (tau)
STDguess=ones(N,1);
Td=linspace(2,3,1000)'; %vector of T1 discritizations
Risk=zeros(N,1);
RISK=zeros(N,1);
W=zeros(N, length(Td)); %matrix of weights
W(1,:)=ones(length(Td),1)*1/length(Td);

Z=100;

for z=1:Z;
display(z)
for j=2:N %iterating through SMC
    
Ttest(j)=expct(Td,W(j-1,:)');
ttest(j)=Ttest(j)/log(2);
Ptest(j)=Pn(ttest(j));
Risk(j-1)=abs(TrueT1-Ttest(j));
Pnew=P(ttest,Ttest(j));
Per=Pnew-Ptest;
STDguess(j)=std(Per(1:j));

    for k=1:length(Td) %Updating weights 
        W(j,k)=normpdf(Ptest(j),P(ttest(j),Td(k)),STDguess(j))*W(j-1,k); 
    end 
W(j,:)=W(j,:)./sum(W(j,:));
end
Risk(end)=abs(TrueT1-Ttest(end));

RISK=RISK+Risk/Z;

end
%plotting stuff because stuff needs to be plotted
h=1/20; %this controls which part of the data gets plotted 
[X,Y]=meshgrid(Td,(N*h:N));
figure(2);
 subplot(2,2,1)
hSurf = surf(X,Y,W(N*h:end,:),'EdgeColor','none','LineStyle','none','FaceLighting','phong');
 subplot(2,2,2)
 plot(h*N:N,RISK(h*N:end));
 subplot(2,2,3)
 plot(STDguess);
% t is tau, T is T1