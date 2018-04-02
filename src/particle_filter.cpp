/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <bits/random.h>
#include <bits/opt_random.h>
#include <bits/random.tcc>

#include "particle_filter.h"

using namespace std;

const double tolerance = 0.000001;// 1e-6
static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	// TODO: what is the right number of particles?
	num_particles = 100;

	normal_distribution<double> nd_x_init(x,std[0]);
	normal_distribution<double> nd_y_init(y,std[1]);
	normal_distribution<double> nd_theta_init(theta, std[2]);

	for (unsigned i = 0; i < num_particles; ++i){
	  Particle p;
	  p.id = i;
	  p.x = nd_x_init(gen);
	  p.y = nd_y_init(gen);
	  p.theta = nd_theta_init(gen);
	  p.weight = 1.0; // setting intial weight to 1

	  particles.push_back(p);
    weights.push_back(1.0);
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> nd_x(0, std_pos[0]);
	normal_distribution<double> nd_y(0, std_pos[1]);
	normal_distribution<double> nd_theta(0, std_pos[2]);

	for (unsigned i = 0; i < num_particles; ++i) {
		// Predict the state for the next time step
		if (fabs(yaw_rate) < tolerance)
		{
			// yaw rate is zero
			particles[i].x += velocity*cos(particles[i].theta)*delta_t;
			particles[i].y += velocity*sin(particles[i].theta)*delta_t;
		} else
		{
			// non zero yaw rate
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta + yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		}

		// Add random Gaussian noise
		particles[i].x += nd_x(gen);
		particles[i].y += nd_y(gen);
		particles[i].theta += nd_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  // Go through the list of observations to figure out the landmark that it is closest to
  for(unsigned o = 0; o < observations.size(); ++o){

    // initialize closest landmark id to -1 (invalid id)
    int closest_landmark_id = -1;

    // initialize distance_squared to this landmark to dBL_MAX
    double min_dist_to_landmark_squared = numeric_limits<double>::max();

    // iterate through all the landmarks and find the closest one
    for(unsigned l = 0; l < predicted.size(); ++l){
      // compute distance squared from current observation to this landmark
      const double current_dist_squared = dist_squared(observations[o].x, observations[o].y, predicted[l].x, predicted[l].y);
      // if computed distance is less than min_distance_to_landmark so far, then set the closet landmark to this
      if (current_dist_squared < min_dist_to_landmark_squared){
        min_dist_to_landmark_squared = current_dist_squared;
        closest_landmark_id = predicted[l].id;
      }
    }

    // Set the observation's id to the closest landmark id
    observations[o].id = closest_landmark_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (unsigned i = 0; i < num_particles; ++i) {

	  const double p_x = particles[i].x;
	  const double p_y = particles[i].y;
	  const double p_theta = particles[i].theta;

	  /***************************************************
	   * Transform the car sensor landmarks from the car coordinate system to the map coordinate system
	  ****************************************************/
	  vector<LandmarkObs> observations_map;
	  for (unsigned o = 0; o < observations.size(); ++o){
	    const int obs_id = observations[o].id;
	    const double obs_x = observations[o].x;
	    const double obs_y = observations[o].y;

	    const double obs_x_map = p_x + cos(p_theta)*obs_x - sin(p_theta)*obs_y;
	    const double obs_y_map = p_y + sin(p_theta)*obs_x + cos(p_theta)*obs_y;
	    observations_map.push_back(LandmarkObs{obs_id, obs_x_map, obs_y_map});
	  }

	  /***************************************************
	   * Only consider the map landmarks within sensor range
	  ****************************************************/
	  vector<LandmarkObs> map_landmarks_in_range;
	  for (unsigned l = 0; l < map_landmarks.landmark_list.size(); ++l){
	    const int landmark_id = map_landmarks.landmark_list[l].id_i;
	    const double landmark_x = map_landmarks.landmark_list[l].x_f;
	    const double landmark_y = map_landmarks.landmark_list[l].y_f;

      // Use dist_squared for efficiency. Avoid square roots!
	    const double current_dist_squared = dist_squared(landmark_x, landmark_y, p_x, p_y);
	    // Extract the landmarks that are within range
	    if (current_dist_squared < (sensor_range*sensor_range))
	      map_landmarks_in_range.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
	  }

	  // Is it possible that map_landmarks_in_range is empty?
	  if (map_landmarks_in_range.empty()){
	    cout << "There are no landmarks in range of this particle!" << endl;
	    particles[i].weight = 0; //assign zero weight to this particle
	    continue; // move on to the next particle
	  }

    /***************************************************
     * Associate the transformed observations to the nearest landmark on the map
    ****************************************************/
	  // clear associations
	  particles[i].associations.clear();
	  particles[i].sense_x.clear();
	  particles[i].sense_y.clear();

    dataAssociation(map_landmarks_in_range, observations_map);


    /***************************************************
     * Update the weight of the particle by applying the multivariate probability density functions
    ****************************************************/
    // reinitialize the particle weight to 1
    double weight = 1.0;
    bool updatedWeightOfParticle = false;
	  // Iterate through all the observations (in map frame)
    for (unsigned om = 0; om < observations_map.size(); ++om){
      // current observation
      const int obs_id = observations_map[om].id;
      const double obs_x = observations_map[om].x;
      const double obs_y = observations_map[om].y;

      // Get the landmark associated with the current observations
      // TODO: Need to find a cleaner way to do this rather than just looping through the list of landmarks to find the associated landmark
      // Would have been better to use a map, but the dataAssociation function needed a vector of map landarks ins range
      // Create a map as well? Or find a way to convert the map into a vector to pass into the data associations function?
      bool foundAssociatedLandmark = false; // set found associated landmark for this observation to false at the start

      double associated_landmark_x, associated_landmark_y;
      for (unsigned l = 0; l < map_landmarks_in_range.size(); ++l){
        if (map_landmarks_in_range[l].id == obs_id){
          associated_landmark_x = map_landmarks_in_range[l].x;
          associated_landmark_y = map_landmarks_in_range[l].y;
          foundAssociatedLandmark = true;

          // set associations
          particles[i].associations.push_back(obs_id);
          particles[i].sense_x.push_back(associated_landmark_x);
          particles[i].sense_y.push_back(associated_landmark_y);
          break;
        }

      }


      // Only calculate and update weights if an associated landmark is found for this observation
      if (!foundAssociatedLandmark){
        cout << "Did not find an associated landmark for observation id: " << obs_id << " particle number:" << i << std::endl;
      }
      else{
        // compute the weight of the particle associated with this observation
        // weight_obs = (1/(2*pi*sigma_landmark_x*sigma_landmark_y))*exp(-(((obs_x - associated_landmark_x)^2/2*sigma_landmark_x^2)+((obs_y - associated_landmark_y)^2/2*sigma_landmark_y^2)))
        const double sigma_landmark_x = std_landmark[0];
        const double sigma_landmark_y = std_landmark[1];
        const double den = 2*M_PI*sigma_landmark_x*sigma_landmark_y;

        // std_x and  std_y are always positive, and so the den has to be always positive, avoiding a divide by zero if std_x or std_y have been set to zero
        if (den > 0.0){
          double weight_obs = exp(-(pow(obs_x - associated_landmark_x, 2)/(2*pow(sigma_landmark_x,2))+ pow(obs_y - associated_landmark_y, 2)/(2*pow(sigma_landmark_y,2))))/den;

          // Combine the probabilities of all the measurements by taking the product
          weight *= weight_obs;
          updatedWeightOfParticle = true;
        } else
        {
          cout << "sigma landmark x or sigma landmark y cannot be set to zero" << endl;
        }
      }
    }

    // If the weight of the particle was not updated, then no associated landmarks were found for this particle!
    // set the weight of this particle to 0.0
    if (!updatedWeightOfParticle){
      cout << "Did not find any landmarks for this particle!" << endl;
      particles[i].weight = 0.0;
      weights[i] = 0.0;
    } else // Otherwise update the particle weight to the calculated weight
    {
      particles[i].weight = weight;
      weights[i] = weight;
      //cout << "Weight of particle" << i << " is " << weight << endl;
    }
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  discrete_distribution<int> dd_index(weights.begin(), weights.end());

  if (particles.size() != num_particles)
    cout << "Error! particles.size is not equal to num_particles";

  vector<Particle> resampled_particles(num_particles);
  for (unsigned i = 0; i < num_particles; ++i){
    const int picked_index = dd_index(gen);

    resampled_particles[i]=particles[picked_index];
  }

  // replace particles with resampled particles
  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
