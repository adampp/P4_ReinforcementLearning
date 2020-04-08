Install/Setup Steps:
1. Download and Extract https://github.com/adampp/P3_UnsupervisedLearning
    https://github.com/adampp/P3_UnsupervisedLearning.git

2. Make sure scikit-learn, numpy, pandas, matplotlib, scipy, gym, and all required dependencies are installed
with Python 3.6+
    pip3 install scikit-learn, numpy, pandas, matplotlib, scipy, gym

3. Additional dependencies included in source, in both ./hiivemdptoolbox and ./forest-env
    run 'pip install -e .' (without quotes)

3. Main executable file is test.py. 
    a. In the comment block, edit problem to be the problem you wish to run, 'forest' or 'frozen'
    b. In the comment block, edit algo to be the algorithm you wish to run, 'pi', 'vi', or 'q'
    
4. To create the various problems used in the paper:
    a. Small Frozen Lake (8x8)  mapsize = 8, and the p in generate_random_map should be 0.9.
    b. Large Frozen Lake (32x32) mapsize = 32, p = 0.96
    c. Small Forest Management edit ./forest-env/gym_forest/envs/forest_env.py default parameter nS=4
    d. Large Forest Management edit ./forest-env/gym_forest/envs/forest_env.py default parameter nS=50