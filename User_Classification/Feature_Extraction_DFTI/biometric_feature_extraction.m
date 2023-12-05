clear
addpath(genpath([pwd filesep 'Output BM']))

%obtain the total number of subject. Note the total subjects should be same
% in all the three folders
fs = 2048;                   %sampling frequency
NSUB = 43; %no of participants
NSESSION = 3;
NGESTURE = 16;              %total number of gestures
NTRIALS = 7;                %total number of trials
a1=[];                      % temporary variable to merge trials of a specific contraction
a2=[];                      % temporary variable to
CompleteSet=[];        % store final gestures and subjects for each session

%% Define output folder
if ~exist('Feature Extracted BM', 'dir')
    mkdir('Feature Extracted BM')
else
    disp('Overwriting')
    rmdir('Feature Extracted BM','s')
    mkdir('Feature Extracted BM')
end

% %% flatten the trials to obtain Gestures x Subjects
% for isession=1:NSESSION
%     for isub =1:NSUB
%         fileName = ['session' num2str(isession) '_participant',num2str(isub),'.mat'];
%         temp_load_forearm=load(fileName,'DATA_FOREARM');
%         datafile=temp_load_forearm.DATA_FOREARM;
%         for igesture = 1:NGESTURE+1        % +1 to include rest gesture
%             for itrial=1:NTRIALS
%                 CompleteSet{isub, igesture, itrial} = datafile{itrial, igesture};
%             end
%         end
%         disp(['Loaded: ' num2str(isession) ' ' num2str(isub)])
%     end
% end
% rmpath(genpath([pwd filesep 'Output BM']))         % to save memory

%% segmentation and processing
count=0;
for isession = 1: NSESSION
    FeatSet={};
    for isub =1:NSUB
        fileName = ['session' num2str(isession) '_participant',num2str(isub),'.mat'];
        temp_load_forearm=load(fileName,'DATA_WRIST');
        datafile=temp_load_forearm.DATA_WRIST;
        for igesture = 1:NGESTURE+1
            for itrial = 1:NTRIALS
                CompleteSet{isub, igesture, itrial} = datafile{itrial, igesture};
                OneSet = CompleteSet{isub,igesture,itrial}';       %shape=16xTotalSamples

                post_process=[];
                for ichannel_2=1:12         % monopolar 8 channel with average referencing
                    temp2 = OneSet(1:12,:);
                    post_process(ichannel_2,:)=OneSet(ichannel_2,:)-mean(temp2,1);
                end
            % %         for ichannel_2=1:8         % bipolar 8 channel processing
            % %             post_process(ichannel_2,:)=OneSet(ichannel_2,:)-OneSet(ichannel_2+8,:);
            % %         end
            % segment the EMG Data
                segData = segmentEMG(post_process', 0.2, 0.15, 5, fs, 1);  % post_process' has to be NSampx8
                %extract the frequency features
                feat= featiDFTl(2048,6,segData);
                FeatSet(isub,igesture,itrial)={feat};
            end
        end
        count=count+1;
        disp(['processed: ',num2str(count),' of ', num2str(NSUB*NSESSION),' files'])
    end
    disp(['saving: Session ',num2str(isession),' biometric data'])
    save(['Feature Extracted BM' filesep 'Forearm_Session' num2str(isession) '.mat'],'FeatSet')
end