

#include <E3d.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <Vision.h>
#include <sstream>
#include <Math/Vector.h>
#include <Math/Matrix.h>
#include <log4cpp/Category.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/SimpleLayout.hh>  
#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/property_tree/ptree.hpp>
#include <Math/ConicOperations.h>




using namespace cv; 

/*
Calculated Distance Invariants for for conics 
Note* : The programme is supposed to take care of both the calculated and the orginal model points 
@param [In] : Center Points < < vec_gt > > or < < vec1 , vec 2 > >
@param [out] : Invariant < <d_gt > >  or < < d1, d2, d3, d4 > > 
*/ 
bool calculateConicDistanceInvariants(const std::vector <std::vector <Point3d> > &centerPoints, std::vector<std::vector<double>> &distanceInvariants);


/*
Calculated Distance Invariants for for conics 
Note* : The programme is supposed to take care 
@param [In] : Normals  < < sn_gt >  > or < < sn1, sn2 > >
@param [out] : Invariant < < theta > >  or < < theta1, theta2, theta3, theta4 > > 
*/ 
bool calculateConicAngleInvariants(const std::vector <std::vector <Point3d> > &surfaceNormals, std::vector<std::vector<double>> &angleInvariants);