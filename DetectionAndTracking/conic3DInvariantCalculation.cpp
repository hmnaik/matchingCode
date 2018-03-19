


#include "conic3DInvariantCalculation.h"

using namespace cv;
using namespace Ubitrack;
using namespace Ubitrack::Vision;
using namespace boost::numeric;
using namespace Ubitrack::Math;
using boost::property_tree::ptree;


using namespace std; 


static log4cpp::Category& logger( log4cpp::Category::getInstance( "conicInvariant" ) );



double getDistance(Point3d vec1 , Point3d vec2){

	double dist = sqrt( pow( vec1.x-vec2.x ,2)+pow( vec1.y-vec2.y,2)+pow(vec1.z-vec2.z,2) ); 
	return dist; 
}

double getAngle(Point3d vec1 , Point3d vec2){

	double angle = (vec1.x*vec2.x + vec1.y*vec2.y + vec1.z*vec2.z) ;
	double norm1 = sqrt (vec1.x*vec1.x + vec1.y*vec1.y + vec1.z*vec1.z) ; 
	double norm2 = sqrt (vec2.x*vec2.x + vec2.y*vec2.y + vec2.z*vec2.z) ; 

	return acos(angle/(norm1*norm2))*180/CV_PI ; 

}



bool calculateConicDistanceInvariants(const std::vector <std::vector <Point3d> > &centerPoints, std::vector<std::vector<double>> &distanceInvariants){

		// To calculate invariant of surface angles and 3D distances we need 2 conics
		// This functions calculates invariant of each point with respect to other points. 
		// That is if we have point {P1,P2...... Pn}
		std::vector<double> pairDistInvariant; 
		int dataValidation = centerPoints.at(0).size();

		for ( int i= 0; i< centerPoints.size(); i++){ // First point 
			//  j = i+1 ; We calculate a upper triangular matrix with all possible relations with minimum computation effort
			for( int j = i+1; j < centerPoints.size() ; j++){ // All next points 

					
				if(dataValidation == 2){

					// dist1 and dist2 is 3D center position provided by our algorithm for a point A ( 1 or them is true)
					Point3d distVec1 = centerPoints.at(i).at(0);
					Point3d distVec2 = centerPoints.at(i).at(1);					
					// dist3 and dist4 is 3D center position provided by our algorithm for a point B ( 1 or them is true)
					Point3d distVec3 = centerPoints.at(j).at(0);
					Point3d distVec4 = centerPoints.at(j).at(1);


					double dist12,dist34;

					// Each point will result in 2 normals and 2 center positions as we know it
					// Therefore for each point A and B, we have 2 normals and 2 center positions each lets say (n1a,n2a and c1a,c2a) and (n1b,n2b and c1b,c2b)
					// each solution leads to having 4 solutions 


					// Distance among the ambiguous solutions , this is expected to remain the same as per literature. 				
					dist12 = getDistance(distVec1,distVec2) ;		
					dist34 = getDistance(distVec3,distVec4) ;		
					LOG4CPP_DEBUG(logger, " Dist between 1-2 : " <<  dist12 <<  " Dist between 3-4 : " <<  dist34 ) ;
					if(dist12 > 1 || dist34 > 1){
						LOG4CPP_ERROR(logger, " Calculated Center points are too distant (> 1mm)") ;
					}

					// Invariant of distance between centers, All 4 solutions are supposed to be constant as per literature. 						

					pairDistInvariant.clear();
					pairDistInvariant.push_back(getDistance(distVec1,distVec3)); // D(1,3)
					pairDistInvariant.push_back(getDistance(distVec1,distVec4)); // D(1,4)
					pairDistInvariant.push_back(getDistance(distVec2,distVec3)); // D(2,3)
					pairDistInvariant.push_back(getDistance(distVec2,distVec4)); // D(2,4)

					distanceInvariants.push_back(pairDistInvariant);

					LOG4CPP_DEBUG(logger, " Dist(1-3): " << pairDistInvariant.at(0) << ":(1-4):" << pairDistInvariant.at(1) 
							<< ":(2-3):" <<  pairDistInvariant.at(2) <<":(2-4): " << pairDistInvariant.at(3)); 
				}
				else if(dataValidation == 1){

					Point3d distVec1 = centerPoints.at(i).at(0);
					Point3d distVec2 = centerPoints.at(j).at(0);

					double dist = getDistance(distVec1,distVec2);
					pairDistInvariant.clear();
					pairDistInvariant.push_back(dist);
					distanceInvariants.push_back(pairDistInvariant);
					LOG4CPP_DEBUG(logger, " GT Distance : " << dist << " Points : " << i << ":" << j);
				}
				else{
					LOG4CPP_ERROR(logger,"Data Input Invalid for distance invariant calculation");
					return FALSE; 
				}
				
			} // Inner loop 
		
		} // Final loop 



	return TRUE; 
}

bool calculateConicAngleInvariants(const std::vector <std::vector <Point3d> > &surfaceNormals, std::vector<std::vector<double>> &angleInvariants){

	// To calculate invariant of surface angles and 3D distances we need 2 conics
	// This functions calculates invariant of each point with respect to other points. 
	// That is if we have point {P1,P2...... Pn}

	std::vector<double> pairDistInvariant; 
	int dataValidation ; 
	dataValidation = surfaceNormals.at(0).size();


	for ( int i= 0; i< surfaceNormals.size(); i++){ // First point 
		//  j = i+1 ; We calculate a upper triangular matrix with all possible relations with minimum computation effort
		for( int j = i+1; j < surfaceNormals.size() ; j++){ // All next points 

			if( dataValidation == 2){
				// Vec1 and Vec2 is solution provided by our algorithm  for a point A ( 1 or them is true) 
				Point3d vec1 = surfaceNormals.at(i).at(0); 
				Point3d vec2 = surfaceNormals.at(i).at(1); 

				// Vec3 and Vec4 is solution provided by our algorithm  for a point B ( 1 or them is true) 
				Point3d vec3 = surfaceNormals.at(j).at(0); 
				Point3d vec4 = surfaceNormals.at(j).at(1); 

				double angle12,angle34;


				// Each point will result in 2 normals and 2 center positions as we know it
				// Therefore for each point A and B, we have 2 normals and 2 center positions each lets say (n1a,n2a and c1a,c2a) and (n1b,n2b and c1b,c2b)
				// each solution leads to having 4 solutions 

				// This is angle among the 2 solutions itself : Not that important as of now 
				//angle12 = getAngle(vec1,vec2); // n1a - n2a
				//angle34 = getAngle(vec3,vec4); // n1b - n2b
				//LOG4CPP_DEBUG(logger, " Angle between 1-2 : " <<  angle12 <<  " Angle between 3-4 : " <<  angle34 ) ;
				
				// Invariant
				pairDistInvariant.clear();
				pairDistInvariant.push_back(getAngle(vec1,vec3));	// n1a - n1b
				pairDistInvariant.push_back(getAngle(vec1,vec4));	// n1a - n2b
				pairDistInvariant.push_back(getAngle(vec2,vec3));	// n2a - n1b
				pairDistInvariant.push_back(getAngle(vec2,vec4));	// n2a - n2b

				angleInvariants.push_back(pairDistInvariant);

				LOG4CPP_DEBUG(logger, " Angle (1-3): " << pairDistInvariant.at(0) << ":(1-4):" << pairDistInvariant.at(1) 
									<< ":(2-3):" <<  pairDistInvariant.at(2) <<":(2-4): " <<  pairDistInvariant.at(3) ); 
			}
			else if(dataValidation == 1){ // if GT data then only one angle to be calculated 
				
				LOG4CPP_DEBUG(logger, " Calculating GT Invatiants for Surface Angles  " );
				// In case of GT we have only one angle  
				Point3d vec1 = surfaceNormals.at(i).at(0); 
				Point3d vec2 = surfaceNormals.at(j).at(0); 
				
				double angle;
				angle = getAngle(vec1,vec2); 
				
				pairDistInvariant.clear();
				pairDistInvariant.push_back(angle);// n1GT - n2GT
				angleInvariants.push_back(pairDistInvariant);

				LOG4CPP_DEBUG(logger, " Angle : " << angle << " :: Points : " << i << ":" << j); 

			} // GT Data : Invariant check end 
			else {
				LOG4CPP_ERROR(logger, " Data Input Invalid for angular invariant calculation  ");
				return FALSE; 
			}
		} // Inner loop 

	} // Final loop 


	return TRUE; 

}


