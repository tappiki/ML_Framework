# retrieve sequential questions using HMM

from yahmm import *
import random
import math

import cx_Oracle

random.seed(261)

def addEmProb(model):

	feature_string = "',".join(enquiry_terms)
	emission_prob_list = [] # list of dictionaries

	for i in range(0,len(states)):
		emission_sql = "select ep.em_prob,ep.states from EMISSION_PROBABILITIES ep where feature_type in (" + feature_string + ") //
						and state = '" + states[i] + "'"
		cursor = execute_query(emission_sql)
		vals = cursor.fetch()
		for j in range(0,len(probs)):
			emmision_dict[vals[j][0]] = vals[j][1]
		emission_prob_list.append(State( DiscreteDistribution(emission_dict) ))
		
		model.add_transition( model.start, emission_prob_list[i], start_probabilities[i] )
		model.add_transition( model.start, model.end, 0.1 )
	return model

states_range = range(0,len(states))

def addTP(model):	

	emission_dict = {}
	for i in state_range:
		transition_range = state_range.pop(i)
		for j in transition_range:
			transition_sql = "select tp.prob from TRANSISTION_PROBABILITIES tp where feature_type = '" + feature_type[i] + "' order by states"
			cursor = execute_query(emission_sql)
			prob = cursor.fetch()
			model.add_transition(emmision_dict[state_range[i],emmision_dict[state_range[i],prob)
	return model

def main():
    # my code here
	
	mdl = Model( name="Question-Prediction" )
	
	global start_probabilities = [ 0.3, 0.2, 0.5]
	
	global states = [ 'capacity' , 'battery_life' , 'brand' , 'lifespan' , 'new'...... ]
	global enquiry_terms = [ 'low_relevance', 'medium_relevance', 'high_relevance' ]
	
	mdl = addEmProb(mdl)
	mdl= addTP(mdl)
	
	model.bake()
	# we use this model to predict the state of next question give previous observation and start
	# we can get the most likely start state through viterbi algorithm
	# The next state is predicted using forward algorithm

if __name__ == "__main__":
    main()
