##### Robocopter 

This is an experimental script using Deep Q Network code from Hands on Machine Learning with Sci-Kit learn and Tensorflow. Using an ARDrone 2.0 and a python client ardrone-python, we are trying to teach the drone to fly by crashing. This was inspired by the research here: http://spectrum.ieee.org/automaton/robotics/drones/drone-uses-ai-and-11500-crashes-to-learn-how-to-fly?imm_mid=0f1a15&cmp=em-data-na-na-newsltr_20170517

TODO:
- [ ] Figure out how to stream video from ARDrone
- [ ] Build functions to process video stream for model
  - [ ] Incorporate processed video stream into tensorflow code
  - [ ] design protocol for crashes and restarting a new flight
  - [ ] design reward heuristic for different manuvers (flying without jerky motions, not hovering for too long)
