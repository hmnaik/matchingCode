
#include <opencv2/opencv.hpp>


#include <Vision/CircularMarkers.h>
#include <Vision/ProcamMarkers.h>

#include <vector>
#include <sstream>
#include <log4cpp/Category.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/SimpleLayout.hh>  

#include <boost/shared_ptr.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <Math/ConicOperations.h>

#include <Util/CalibFile.h>
#include <Util/GlobFiles.h>

#include <tinyxml.h>
#include <Vector3D.h>

#include <Calibration/AbsoluteOrientation.h>
#include <Calibration/2D3DPoseEstimation.h>
#include <Calibration/PoseEstimation3D3D.h>
#include <Calibration/2D3DPointMatching.h>

#include <Vision\Debugging.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <Math/conic3DInvariantCalculation.h>

#include <Util/GlobFiles.h>


#define DEBUG_WINDOWS 0
#define DEBUG_IMAGE_ACTIVE 0
// #define UNDISTORT_IMAGE 
#define DEBUG_INVARIANT_DATA 0


using namespace cv;
using namespace Ubitrack;
using namespace Ubitrack::Vision;
using namespace boost::numeric;
using namespace Ubitrack::Math;
using boost::property_tree::ptree;
using namespace std; 

int mx = 100;
int my = 100;

static log4cpp::Category& logger( log4cpp::Category::getInstance( "imageDetection" ) );

// Flip the Y coordinate of the points to match with the GT data 
void flipY( std::vector< Ubitrack::Math::Vector < 2 > > &midPoints, int Width){

	for(int i = 0; i < midPoints.size(); i++ )
		midPoints.at(i)(1) = Width -1 - midPoints.at(i)(1); 

} 

void my_mouse_callback( int event, int x, int y, int flags, void* param ){
	switch( event ){
	case CV_EVENT_LBUTTONDOWN:
		mx = x;
		my = y;
		std::cout << "X: "<<x<<" Y: "<<y<<"\n";
		break;
	}
}

bool loadCalibSettingsFromXml( std::string filePath, Math::Matrix3x3 &intrinsicMatrix )
{
	TiXmlDocument settingsFile;
	settingsFile.LoadFile ( filePath );

	if( settingsFile.Error() ){
		LOG4CPP_WARN( logger, "Could not open XML file for loading expert settings.");
		return false;
	}

	TiXmlNode * cameraElement = settingsFile.FirstChildElement( "Camera" );
	TiXmlElement * matrixElement = settingsFile.FirstChildElement( );
	//loadAlgorithmsSettingsFromXML ( algorithmsSettingsElem, algorithmsSettings );
	//loadCircularMarkerSettingsFromXML ( circularMarkerSettingsElem, markerSettings );
	return true;
}


Rect safeRect(Rect r, int width, int height)
{
	r.x = max(0, min(width-1, r.x));
	r.y = max(0, min(height-1, r.y));

	if (r.width+r.x > width)
		r.width = width - r.x - 1;

	if (r.width+r.y > height)
		r.height = height - r.y - 1;

	return r;
}


void showDebugWindows(Mat imgGray, Mat imgBin, Mat imgDbg, std::string name, int width, int height, int posX, int posY) {
	Rect r = safeRect(Rect(posX-width/2,posY-height/2,width, height), imgGray.cols, imgGray.rows);
	if (r.width == 0 || r.height == 0)
		return;
	stringstream ss;
	ss << name;
	if (!imgGray.empty()) {
		Mat roi = Mat(imgGray, r);
		ss << " Gray";
		namedWindow ( ss.str(), CV_GUI_EXPANDED );

		imshow(ss.str().c_str(), roi);
	}
	if (!imgBin.empty()) {
		Mat roi = Mat(imgBin, r);
		ss << " Bin";
		namedWindow ( ss.str(), CV_GUI_EXPANDED );

		imshow(ss.str().c_str(), roi);
	}
	if (!imgDbg.empty()) {
		Mat roi = Mat(imgDbg, r);
		ss << " Dbg";
		namedWindow ( ss.str(), CV_GUI_EXPANDED );

		imshow(ss.str().c_str(), roi);
	}
}

bool verifyTriplet(const std::vector<std::vector<int>> &indexMatching,
	const std::vector <std::vector <int> > &calculatedPointPair,
	const std::vector <std::vector <int> > &modelPointPair, 
	const Math::Vector<3> &modelTriplet,
	const Math::Vector<3> &imageTriplet,
	int indexStatus = 0){



		int modelPt2, modelPt3 ; 
		modelPt2 = modelTriplet[1];
		modelPt3 = modelTriplet[2];
		int imagePt2, imagePt3 ; 
		imagePt2 = imageTriplet[1]; 
		imagePt3 = imageTriplet[2]; 

		for (int i = indexStatus ; i < indexMatching.size() ; i++ ){

			int modelPairID = indexMatching.at(i).at(0); // Represents an ID of model point pair 
			int imagePairID = indexMatching.at(i).at(1); // Represents corresponding ID of image point pair
			int modelPoint1 =  modelPointPair.at(modelPairID).at(0); 
			int modelPoint2 = modelPointPair.at(modelPairID).at(1) ; 
			if( (modelPoint1== modelPt2) && (modelPoint2 == modelPt3) ) {

				if ( ( (calculatedPointPair.at(imagePairID).at(0) == imagePt2) && (calculatedPointPair.at(imagePairID).at(1) == imagePt3) ) 
					|| ( (calculatedPointPair.at(imagePairID).at(1) == imagePt2) && (calculatedPointPair.at(imagePairID).at(0) == imagePt3) ) ){
						return true; 
				}
			} 	

		} 


		return false; 
} 

void filterPointPair( const std::vector<std::vector<int>> &indexMatching, 
	const std::vector <std::vector <int> > &calculatedPointPair,
	const std::vector <std::vector <int> > &modelPointPair,
	std::vector<std::vector<int>> &modelImagePair, 
	multimap<int ,int > &matchingMap,
	std::vector<std::vector<Math::Vector3>>  &matchingTriplet,
	Mat &matchingMat){

		for(int i = 0; i < indexMatching.size() ; i++){
			int model = indexMatching.at(i).at(0); 
			int image = indexMatching.at(i).at(1); 
			LOG4CPP_DEBUG(logger, " " << model <<  ": Model Pair " << modelPointPair.at(model).at(0) << "," << modelPointPair.at(model).at(1)
				<< " " << image << " : Image Pair " << calculatedPointPair.at(image).at(0) << "," << calculatedPointPair.at(image).at(1) 
				<< " : Normal " << indexMatching.at(i).at(2) ); 
		}
		int modelPairID, imagePairID,nextModelPairID,nextImagePairID; 
		for (int i = 0 ; i < indexMatching.size(); i++){ // For each pair checking 

			Math::Vector<3> modelTriplet; 
			Math::Vector<3> imageTriplet; 
			std::vector<Math::Vector3> triplet; 

			for (int j = i+1 ; j < indexMatching.size() ; j ++ ) {

				modelPairID = indexMatching.at(i).at(0); // Represents an ID of model point pair 
				imagePairID = indexMatching.at(i).at(1); // Represents corresponding ID of image point pair
				nextModelPairID = indexMatching.at(j).at(0); // Represents an ID of next model point pair 
				nextImagePairID = indexMatching.at(j).at(1); // Represents an ID of next image point pair  

				int modelPt1, modelPt2, modelPt3, imagePt1, imagePt2 , imagePt3; 
				// If either of model or Image IDs are same then matching has no point
				// ie if Model ID  0(pt1,pt2) maps to two different image pair ID say 1(i1,i2) and 2(i1',i2') we wont be able to conclude matching result 
				// same if a same image pair matches with 2 model pair we are left with ambiguity. 
				if( (modelPairID != nextModelPairID) && (imagePairID != nextImagePairID) ) {
					// Check if first element of first and second pair is same ( other two will never be same , algorithm constrains as 
					// order is set up in ascending order of model points ) 
					// How it helps 
					// e.g. 0( 1005,1008) & 1(1005,1009) bin 00  can be only type possible : Verification pair 1008, 1009  
					if( (modelPointPair.at(modelPairID).at(0) == modelPointPair.at(nextModelPairID).at(0) ) ){ // bin 00  							
						modelPt1 = modelPointPair.at(modelPairID).at(0); // common 
						modelPt2 = modelPointPair.at(modelPairID).at(1);
						modelPt3 = modelPointPair.at(nextModelPairID).at(1);

						// e.g. 0 (1 ,2)  &  1 ( 1, 3)  Lets say binary 00 
						if( (calculatedPointPair.at(imagePairID).at(0) == calculatedPointPair.at(nextImagePairID).at(0) )) { 

							imagePt1 = calculatedPointPair.at(imagePairID).at(0); // common first 
							imagePt2 = calculatedPointPair.at(imagePairID).at(1);
							imagePt3 = calculatedPointPair.at(nextImagePairID).at(1);
						}
						// e.g. 0 ( 1,2) & 1 ( 0 ,1 ) Lets say binary 01 
						else if(	(calculatedPointPair.at(imagePairID).at(0) == calculatedPointPair.at(nextImagePairID).at(1) )	) {

							imagePt1 = calculatedPointPair.at(imagePairID).at(0); // common first 
							imagePt2 = calculatedPointPair.at(imagePairID).at(1);
							imagePt3 = calculatedPointPair.at(nextImagePairID).at(0);
						}
						// e.g. 0 ( 1,2) & 1 ( 2 , 3  ) Lets say binary 10 
						else if(	(calculatedPointPair.at(imagePairID).at(1) == calculatedPointPair.at(nextImagePairID).at(0)	)	) {

							imagePt1 = calculatedPointPair.at(imagePairID).at(1); // common first 
							imagePt2 = calculatedPointPair.at(imagePairID).at(0);
							imagePt3 = calculatedPointPair.at(nextImagePairID).at(1);
						}
						// e.g. 0 ( 1, 4) & 1 ( 3 , 4 ) Lets say binary 11 
						else if( (calculatedPointPair.at(imagePairID).at(1) == calculatedPointPair.at(nextImagePairID).at(1) ) ) {

							imagePt1 = calculatedPointPair.at(imagePairID).at(1);  // common point 
							imagePt2 = calculatedPointPair.at(imagePairID).at(0);
							imagePt3 = calculatedPointPair.at(nextImagePairID).at(0);
						}

						else {

							LOG4CPP_DEBUG(logger, " Pairs considered for image points have no point in common "); 
							continue; 
						}


						// Clean the last triplet 
						modelTriplet.clear();
						imageTriplet.clear();
						// Save new triplet in the vector 
						modelTriplet[0] = (modelPt1);
						modelTriplet[1] = (modelPt2);
						modelTriplet[2] = (modelPt3);
						imageTriplet[0] = (imagePt1); 
						imageTriplet[1] =(imagePt2); 
						imageTriplet[2] =(imagePt3); 

						// verification pair would only be present after primary pair and secondary pair 
						// i = 0 ( 1005, 1008 ) and j = 5 (1005,1010) now verification pair K = (1008,1010) woulbe be obviously after j  
						int indexStuatus = 0; 
						indexStuatus = j +1 ; 
						if( verifyTriplet(indexMatching, calculatedPointPair, modelPointPair, modelTriplet, imageTriplet , indexStuatus) ) {

							LOG4CPP_DEBUG(logger, " Model triple : " << modelTriplet << ": Image triple : " << imageTriplet); 

							// **************** Triplets & Multimap: Begin, Not used as of Now ****************
							triplet.clear(); // Clear the triplet 
							triplet.push_back( modelTriplet) ; // Save model triplet 
							triplet.push_back(imageTriplet); // Save image triplet 
							// Saving the whole triplet 
							matchingTriplet.push_back(triplet); 

							// Insert all possible pair combination - Matrix does the job
							/*matchingMap.insert(std::pair<int,int>( imagePt1,modelPt1 ));
							matchingMap.insert(std::pair<int,int>( imagePt2,modelPt2 ));
							matchingMap.insert(std::pair<int,int>( imagePt3,modelPt3 ));*/

							//*************** Triplet & Multimap : End **************************

							// Filling up the voting matrix 
							matchingMat.at<double>(imagePt1,modelPt1) = matchingMat.at<double>(imagePt1,modelPt1)+1;
							matchingMat.at<double>(imagePt2,modelPt2) = matchingMat.at<double>(imagePt2,modelPt2)+1;
							matchingMat.at<double>(imagePt3,modelPt3) = matchingMat.at<double>(imagePt3,modelPt3)+1;

						} // Triplet Verification 


					} // assign model and image points for triplet 

					// e.g. 0( 1005,1008) & 1(1004,1005) bin 01, not possible due to ascending scheme  
					// e.g. 0( 1005,1008) & 1(1008,10013) bin 10 , 1008 is common but verification pair is 1005/1013 which has already been considered 
					// e.g. 0( 1005,1008) & 1(1006,1008) bin 11, 1008 is common but verification pair is 1005/1006 which has been considered 
					else{
						LOG4CPP_DEBUG(logger, " First element of given pairs is not same, common point must be first  " ); 
					}

				} // model Id Check 
				else {		
					LOG4CPP_DEBUG(logger, " Same model or Image point pair!! Cant generate triplet " );			
				}


			}



		} 




}


void createCorrespondenceMap( multimap<int ,int > &matchingMap, Mat& votingMatrix,  Mat& weightedMat){


	int modelPointSize= votingMatrix.cols ; 
	int imagePointSize = votingMatrix.rows ; 

	// Now clear the map 
	matchingMap.clear();

	int votingSelectionThreshold =  *max_element(votingMatrix.begin<double>(),votingMatrix.end<double>());
	votingSelectionThreshold /= 2; 
	LOG4CPP_DEBUG(logger,  "Voting Selection threshold : " << votingSelectionThreshold);

	for(int i = 0 ; i < votingMatrix.rows; i++ ){ // Selection row component for itration 

		int minVoteCount = 1; 
		int matchingIndex = 0; 
		int zero = 0; 
		bool matchFound = false; 

		LOG4CPP_DEBUG(logger, " New row : " << i );

		double maxRowVote;
		Point maxRowInd;
		// Taking the row from matrix 
		Mat rowMatrix = votingMatrix.row(i).clone();
		// Find max voting elelemt and its position 
		// i.e  0 1 2 3 4 5 X 0 0 -> maxVal = X and pos is (6,1)
		minMaxLoc(rowMatrix, NULL , &maxRowVote, NULL,&maxRowInd); 
		// Checking existance of such maxima (To know if its unique or ambiguous) 
		int maxRowCount = std::count(rowMatrix.begin<double>(),rowMatrix.end<double>(),maxRowVote);
		LOG4CPP_DEBUG(logger, "Position:  max indices " << maxRowInd.y << " - " << maxRowInd.x << ",  val : " << maxRowVote << ", Exisance " << maxRowCount);

		double maxColVote;
		Point maxColInd;
		//Find maxima in particular column where row maxima exists , i.e col = xPos, (6) as per above example
		Mat colMatrix = votingMatrix.col(maxRowInd.x).clone();
		// Find maxima in column 
		// Its a column matrix : [ 0 1 2 X 0 0 0 ]' , Maxima = X , Pos= (1,3)
		minMaxLoc(colMatrix, NULL , &maxColVote , NULL , &maxColInd); 
		// Count no of such maxima (To know if its unique or ambiguous) 
		int maxColCount = std::count(colMatrix.begin<double>(),colMatrix.end<double>(),maxRowVote);
		LOG4CPP_DEBUG(logger, "Position:  max indices " << maxColInd.y << " - " << maxColInd.x << ",  val : " << maxColVote << ", Existance" << maxColCount);

		int rowNos,colNos; 
		// Row position of our requiered point 
		rowNos = maxColInd.y; 
		// Column position of our required point 
		colNos = maxRowInd.x; 

		// If the matching is unique, max element found in row and column is only one position 
		// and both row and column have only one existance of such vote , that wold mean this matching is unique among all matching matrix 

		if( i == rowNos && maxRowVote == maxColVote && maxRowCount==1 && maxColCount==1){

			if(  maxRowVote > votingSelectionThreshold ){

				matchingMap.insert(std::pair<int,int>(rowNos,colNos));
				weightedMat.at<double>(i,matchingMap.find(i)->second) = 1;	

			} 


		}
	}

	LOG4CPP_TRACE(logger," Weighted Matrix \n " << weightedMat );


}


void getPairCombinations(const std::vector <int> &pointIDs, std::vector< std::vector <int> > &pairCombination){

	int noOfPoints = pointIDs.size(); 
	std::vector<int> tempPair; 

	for(int i =0 ; i < noOfPoints ; i++ ){

		for(int j = i+1 ; j < noOfPoints ; j++ ){

			tempPair.clear();
			tempPair.push_back(pointIDs.at(i)); 
			tempPair.push_back(pointIDs.at(j)); 
			pairCombination.push_back(tempPair); 

		}


	}

}

void getPairCombinations(int pointIDSize, std::vector< std::vector <int> > &pairCombination){


	std::vector<int> tempPair; 

	for(int i =0 ; i < pointIDSize ; i++ ){

		for(int j = i+1 ; j < pointIDSize ; j++ ){

			tempPair.clear();
			tempPair.push_back(i); 
			tempPair.push_back(j); 
			pairCombination.push_back(tempPair); 

		}


	}

}

// passing reference of vector makes the job much faster 
bool findMatchingPointPairs(const std::vector<std::vector<double> >  &calculatedDistInvariants,
	const std::vector<std::vector<double> > &calculatedAngleInvariants,
	const std::vector<std::vector<double> >  &modelDistInvariants,
	const std::vector< std::vector<double> > &modelAngleInvariants,
	std::vector<std::vector <int> > &indexMatching,
	double distMatchingThresh = 20,
	double angleMatchingThresh = 5 ){


		if( (calculatedDistInvariants.size() != calculatedAngleInvariants.size() ) || 
			( modelDistInvariants.size() != modelAngleInvariants.size() ) ){

				LOG4CPP_ALERT(logger, "Size of the Invariants don't match !! Wathcout. ");

		}
		LOG4CPP_DEBUG(logger, "Matching Results : \n "); 
		std::vector<int> matchingPair; 

		for(int i = 0 ; i < modelDistInvariants.size() ; i ++ ){ // Loop for all model Point invariants 

			for(int j = 0 ; j < calculatedDistInvariants.size() ; j ++ ){ // loop for all calculated point invariants 

				// Check for matching distance 
				double distAvg = calculatedDistInvariants.at(j).at(0)+calculatedDistInvariants.at(j).at(1)+
					calculatedDistInvariants.at(j).at(2) + calculatedDistInvariants.at(j).at(3); 
				distAvg = distAvg/4; 
				double distDiff = abs((modelDistInvariants.at(i).at(0) - distAvg) ); 

				if( distDiff < distMatchingThresh ) {

					bool foundMatch = false; 
					double lastDiff = 100; 
					int normalIndex = 0; 
					for(int k = 0 ; k < calculatedAngleInvariants.at(j).size(); k++){

						double angleDiff = abs( modelAngleInvariants.at(i).at(0) - calculatedAngleInvariants.at(j).at(k) ) ; 		
						if( angleDiff < angleMatchingThresh && angleDiff < lastDiff){

							lastDiff = angleDiff ; 
							normalIndex = k ; 
							foundMatch = true; 

						} // taking minimum
					} // loop for angle matching withing 4 invariants 
					if( foundMatch == true) {
						matchingPair.clear(); 
						matchingPair.push_back(i); // Model Point Index 
						matchingPair.push_back(j); // Calculated Point Index 
						matchingPair.push_back(normalIndex); // Normal 
						indexMatching.push_back(matchingPair); // Saving the whole match 
						LOG4CPP_DEBUG(logger, " Model : " << i << " Calculated " << j << " Normal : " << normalIndex ); 
					}

				} // loop end for distance matching 


			} // loop end for all calculated point iteration 

		} // loop end for all model point interation 



		return TRUE; 

}


void initLogging()
{
	log4cpp::Appender* app = new log4cpp::OstreamAppender( "stderr", &std::cerr );
	log4cpp::SimpleLayout * layout = new log4cpp::SimpleLayout();
	app->setLayout( layout );

	log4cpp::Category::getRoot().setAdditivity( false );
	log4cpp::Category::getRoot().addAppender( app );
	log4cpp::Category::getRoot().setPriority( log4cpp::Priority::INFO ) ; // default: INFO
	log4cpp::Category::getInstance( "Ubitrack.Events" ).setPriority( log4cpp::Priority::NOTICE ); // default: NOTICE
}

/**
* Project 3d point (ubitrack format) to image 2d point ( opencv format )
*/
cv::Point project( Math::Matrix < 3, 3 > intrinsic, Math::Vector < 3 > v, int imageHeight) {
	Math::Vector < 3 > projected = ublas::prod ( intrinsic, v );
	projected /= projected(2);
	return cv::Point ( static_cast< int > ( projected(0) ), static_cast< int > ( /* imageHeight - 1 - */ projected(1) ) );
}

bool loadModelPoints(std::string filePath, 
	std::vector<std::vector<Point3d>> &modelData, 
	std::vector<std::vector<Point3d>> &modelPointNormals, 
	std::vector<int> &modelIDIndex,
	std::vector<double> &markerSize) {

		std::string line;
		std::ifstream file( filePath );
		// Could work without nested vector as well , since we have only one center,normal for model points 
		// To be compatible with the invariant calculation and writing file with same method 
		std::vector<Point3d> point3dData;
		std::vector<Point3d> normal3dData;

		std::locale origLocale = std::locale::global(std::locale::classic());
		LOG4CPP_DEBUG( logger, "Parsing MarkerCircular file: "  << std::string( filePath ));
		std::string dataFileExtFormat = ".txt"; 

		if( dataFileExtFormat.compare(boost::filesystem3::extension(filePath.data())) != 0) {
			return FALSE; 
		}


		if (file.is_open()) {
			while (file.good()) {
				getline( file, line );

				LOG4CPP_DEBUG( logger, "Parsing line: "  << line);

				std::vector< std::string > tokenList;
				boost::split( tokenList, line, boost::is_any_of("\t "), boost::token_compress_on);
				if (tokenList.size() < 8) continue;


				try {
					LOG4CPP_DEBUG( logger, "ID: "  << tokenList[0] << " \n Vector: " << tokenList[1] << "; " << tokenList[2] << "; " << tokenList[3] << "; " <<
						"\n Normal: "  << tokenList[4] << "; " << tokenList[5] << "; " << tokenList[6] << "; " << "\n Radius : " << tokenList[7]);


					modelIDIndex.push_back(boost::lexical_cast<int>(tokenList[0]));
					double x = boost::lexical_cast<double>(tokenList[1]);
					double y = boost::lexical_cast<double>(tokenList[2]);
					double z = boost::lexical_cast<double>(tokenList[3]);
					double nx = boost::lexical_cast<double>(tokenList[4]);
					double ny = boost::lexical_cast<double>(tokenList[5]);
					double nz = boost::lexical_cast<double>(tokenList[6]);
					double size = boost::lexical_cast<double>(tokenList[7]); 
					markerSize.push_back(size);

					point3dData.clear();
					normal3dData.clear();

					point3dData.push_back(Point3d(x,y,z));
					normal3dData.push_back(Point3d(nx,ny,nz));

					modelData.push_back(point3dData);
					modelPointNormals.push_back(normal3dData);

				} catch (std::exception& e) {
					LOG4CPP_DEBUG( logger, "Exception found, catched: " << e.what());
					return FALSE;
				}
			}

			if( count(markerSize.begin(),markerSize.end(),markerSize.at(0)) != markerSize.size() ){
				LOG4CPP_ALERT(logger,"HEY!! 3D model Data has multiple Radius, watchout "); 
			}


			file.close();



			return TRUE; 

		}
}


bool readSettingsFile( std::string &filePath, CIRCULAR_MARKER_SETTINGS &markerDetectionSettings){


	// Load settings from file 
	TiXmlDocument settingsFile;
	std::string SettingsFilePath = filePath.data(); 	
	SettingsFilePath.append("/settings.xml");
	settingsFile.LoadFile ( SettingsFilePath );

	if( settingsFile.Error() ){
		cout << "couldn't open the xml file with settings" << endl;
		return false;
	}

	TiXmlElement * circularMarkerSettingsElem = settingsFile.FirstChildElement("DetectionSettings");
	loadCircularMarkerSettingsFromXML ( circularMarkerSettingsElem, markerDetectionSettings );

	return true; 

}

bool writeInvariantFile(std::string &fileName, 
	const std::vector <std::vector <double >> &distInvariants, 
	const std::vector <std::vector <double >> &angleInvariants, 
	double noOfPoints){

		ofstream outfile; 

		outfile.open(fileName,ios::out); 

		if (!outfile) {
			cerr << "Can't open output file for Header " << endl;
			exit(1);
		}
		if(distInvariants.size() == 0 || angleInvariants.size()==0 || noOfPoints == 0){
			
			return false ; 
		}


		if( distInvariants.at(0).size() == 1 && angleInvariants.at(0).size() == 1){

			outfile << "Point 1 , Point 2 ," << " Angle, "<< " Distance " << endl ;  

			double n = 0; 
			for ( int i= 0; i< noOfPoints; i++){ // First point 
				//  j = i+1 ; We calculate a upper triangular matrix with all possible relations with minimum computation effort
				for( int j = i+1; j < noOfPoints; j++){ // All next points 

					LOG4CPP_DEBUG(logger, "Betweens Points  : " << i << " And " << j  );

					// Writing Point Infor in the csv file 
					outfile << i << "," << j << ","; 

					outfile << angleInvariants.at(n).at(0) << ", " << distInvariants.at(n).at(0) << endl ; 
					n++;
				}
			}
		}
		else if(distInvariants.at(0).size() == 4 && angleInvariants.at(0).size() == 4){

			outfile << "Point 1 , Point 2 ," << "Ang 1, Ang 2 , Ang 3 , Ang 4 , "<< "dist1 , dist 2 , dist 3 , dist 4" << endl ;  

			// To calculate invariant of surface angles and 3D distances we need 2 conics
			// This functions calculates invariant of each point with respect to other points. 
			// That is if we have point {P1,P2...... Pn}
			double n=0; 
			for ( int i= 0; i< noOfPoints; i++){ // First point 
				//  j = i+1 ; We calculate a upper triangular matrix with all possible relations with minimum computation effort
				for( int j = i+1; j < noOfPoints ; j++){ // All next points 

					LOG4CPP_DEBUG(logger, "Betweens Points  : " << i << " And " << j  );

					// Writing Point Infor in the csv file 
					outfile << i << "," << j << ","; 

					outfile << angleInvariants.at(n).at(0) << ", " << angleInvariants.at(n).at(1) << "," << angleInvariants.at(n).at(2) << "," << angleInvariants.at(n).at(3)  << " , "; 
					outfile << distInvariants.at(n).at(0) << ", " << distInvariants.at(n).at(1) << "," << distInvariants.at(n).at(2) << "," << distInvariants.at(n).at(3) << ", " << endl ; 
					n++;
				}
			}

		}
		else{

			return FALSE; 
		}
		outfile.close();

		return TRUE; 
}

void drawPoseE3D ( cv::Mat & dbgImage, Math::Pose pose, Math::Matrix < 3, 3> projection, int thickness, double error, double minError, double maxError )
{
	// the corner points
	static float points3D[ 4 ][ 3 ] = {
		{ 0.0f, 0.0f, 0.0f }, { 1.0f, 0.0f, 0.0f }, { 0.0f, 1.0f, 0.0f }, { 0.0f, 0.0f, 1.0f },
	};

	// project points
	Math::Vector< 3, double > p2D[ 4 ];
	for ( int i = 0; i < 4; i++ )
	{
		p2D[ i] = projectPoint ( pose * (Math::Vector< 3, float >( points3D[ i ] ) * 20.0), projection, dbgImage.rows);
	}

	CvScalar colors[3] = {CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), CV_RGB(0, 0, 255)};

	// draw circle representing uncertainty
	cv::circle ( dbgImage, cvPoint ( p2D[0](0), p2D[0](1)), cvRound( dbgImage.cols / 50.0 ),
		cv::Scalar(error, minError, maxError), thickness); // Ubitrack::Vision::getGradientRampColor (error, minError, maxError)

	// draw some lines
	for ( int i = 1; i < 4; i++ ) {
		cv::line ( dbgImage, cvPoint ( p2D[0](0), p2D[0](1)), cvPoint ( p2D[i](0), p2D[i](1)),
			colors[i-1], thickness);
	}
}

/*
Internal Function 
Projects 3d point on 2d image plane using projection Matrix P = K [R|T]
Returns dehomogenised 2D image point 

*/ 
Math::Vector2 project3d (const Math::Matrix<3,4>& projectionMatrix, Math::Vector3 & point3d ) 
{
	// x = P*X ( P = 3x4 projection Matrix , X = 4x1 Homogenous 3D point, x = 3x1 homogenous 2d point) 
	Math::Vector < 3 > projected = ublas::prod ( projectionMatrix ,Math::Vector<4> (point3d[0],point3d[1],point3d[2],1) );
	// Converting 2d point from form [x y w] -> [x/w y/w 1]
	projected /= projected(2);

	return Math::Vector2(projected(0),projected(1));
}

void match2d3dpoints( const std::vector < Math::Vector2 > &points2d , const std::vector< Math::Vector3 >& points3d ,	
	const Math::Matrix<3,4>& projectionMatrix , multimap<int,int> & matchingMap , double minMatchDistance){

		multimap<int,int> tempMap; 
		tempMap = matchingMap; 
		matchingMap.clear(); 
		for (int i = 0 ; i < points3d.size() ; i++){ // loop of point3d 

			// Projecting each point 

			Math::Vector2 imagePoint = project3d(projectionMatrix, (Math::Vector <3>)points3d.at(i) ); 

			LOG4CPP_INFO(logger, " " << i << " : "  << imagePoint );

			double minDist = 100 ; 
			int minIndex = -1; 

			for(int j=0 ; j < points2d.size(); j++) { // loop through all detected 2d points 

				double dist = boost::numeric::ublas::norm_2(points2d.at(j) - imagePoint) ; 

				if(dist < minDist && dist < minMatchDistance ){

					minDist = dist;
					minIndex = j ; 

				}


			}

			if( minIndex != -1 ){

				// If the map already exists then we don't need it again 
				if( tempMap.find(minIndex) == tempMap.end() ) {

					matchingMap.insert(std::pair<int,int> (minIndex,i) ); 
					LOG4CPP_INFO(logger, " " << minIndex << " : "  << points2d.at(minIndex) << " : " << i << " " << imagePoint << " - Dist " << minDist);
				}
			}


		}






}





int main( int argc, char ** argv )
{
	initLogging();

	Mat imgGray, imgDbg, imgTmp;


	bool useCamera = false;
	bool detect = false; 
	bool useBarcodeScanner = false; // HMN - Barcode on off 

	bool show_debug_window = DEBUG_WINDOWS;
	bool show_image_active = DEBUG_IMAGE_ACTIVE; 

	std::string filePath, data3DFileName ; 
	CvCapture* capture; 



	/*filePath = argv[0]; 
	data3DFileName = argv[1]; */

	
	// filePath = "D:/Extend3D/Images/CircularBoard/4CircleBoard"; data3DFileName = "/3DData_9.txt";
	// filePath = "D:/Extend3D/Images/CircularBoard/MultipleMarkerNonSymmetric"; data3DFileName = "/3DData_9.txt";
	
	// Car Model Data Set 
	 filePath = "D:/Extend3D/Images/CarData12";  data3DFileName = "/CarModel3DData_12.txt";
	// filePath = "D:/Extend3D/Images/CarData5_8/5_modified"; data3DFileName = "/CarModel3DData_5.txt"; 		
	// filePath = "D:/Extend3D/Images/CarData5_8/8_modified"; data3DFileName = "/CarModel3DData_8.txt";

	//Web Cam Data 
	 // filePath = "D:/Extend3D/Images/CarData5_8/WebCamData_5"; data3DFileName = "/CarModel3DData_5.txt"; 
	 // filePath = "D:/Extend3D/Images/CarData5_8/WebCamData_8"; data3DFileName = "/CarModel3DData_8.txt"; 
	// filePath = "D:/Extend3D/Images/CarData12/WebCamData_12";  data3DFileName = "/CarModel3DData_12.txt";

	
	std::string fileName;
	std::string invDataFileName , debugImageName , debugPath, undistImgPath, undistImgName; 	
	std::vector < std::vector < Point3d > > modelPoints,modelPointNormals, detectedPoints;
	std::vector <int> modelPointIDs; 
	std::vector <double> markerSize; 
	std::list < boost::filesystem3::path > files;
	std::list < boost::filesystem3::path >::iterator fileIt;
	int imageIndex = 0; 
	int noOfFiles = 0; 

	if(useCamera){

		// VideoCapture
		capture = cvCaptureFromCAM( CV_CAP_ANY );
		
		if ( !capture) {
			fprintf( stderr, "ERROR: capture is NULL \n" );
			getchar();
			return -1;
		}

	}
	else{

		boost::filesystem::path dtbFile (filePath);
		if (!boost::filesystem::exists( dtbFile ) || !boost::filesystem::is_directory( dtbFile )) {
			LOG4CPP_ERROR(logger, "Database not found!" ) ;
			return 0;
		}
		LOG4CPP_INFO(logger, "Globbing files from directory...");
		Ubitrack::Util::globFiles (filePath, Util::FilePattern::PATTERN_OPENCV_IMAGE_FILES, files );
		fileIt = files.begin();
		noOfFiles = files.size();
	}

	// unsigned int cameraId = 0;

	// Creating folder for debug Information in (debugData Folder in the Path) 
	debugPath = filePath.data(); 
	debugPath.append("/debugData");
	boost::filesystem3::path debugDir(debugPath);

	// If directory exists then remove existing data 
	if( boost::filesystem3::exists(debugDir)) {
		// Remove old directory 
		if(!boost::filesystem3::remove_all(debugDir))
			LOG4CPP_ERROR(logger, "Sorry!! Could not delete existing debug directory");
	}

	// Create new one ( Creating a fresh directory on each exicution ) 
	if(!boost::filesystem3::create_directory(debugDir))
		LOG4CPP_ERROR(logger, "Sorry!! Could not create debug directory");

	// Undistorting and Saving the image in the file path : newName -> undistImageFILENAME
#ifdef UNDISTORT_IMAGE
	undistImgPath = filePath.data();
	undistImgPath.append("/undistImage");
#endif


	// Reading Data from 3D File 		
	std::string model3DDataFile, modelInvariantFile; 
	model3DDataFile = filePath.data(); 
	model3DDataFile.append(data3DFileName);
	std::vector < std::vector <double> > modelDistInvariants;
	std::vector < std::vector <double> > modelAngleInvariants;
	std::vector <std::vector <int>> modelPointPair; 
	// Loading 3D points ; Note : MarkerSize is in Diameters, Use Radius 
	if( loadModelPoints(model3DDataFile,modelPoints,modelPointNormals,modelPointIDs, markerSize) ){

		getPairCombinations(modelPointIDs.size(),modelPointPair); // Get combination with mapped IDs 
		// getPairCombinations(modelPointIDs,modelPointPair);  // Get combination with natural ID 

		if(! calculateConicDistanceInvariants(modelPoints, modelDistInvariants)){
			LOG4CPP_ERROR(logger,"Calculating Invariants Failed");
		} 
		if( !calculateConicAngleInvariants(modelPointNormals,modelAngleInvariants)) {
			LOG4CPP_ERROR(logger,"Calculating Invariants Failed");
		} 

		if(DEBUG_INVARIANT_DATA){
			modelInvariantFile = filePath.data();
			modelInvariantFile.append("/modelInvariantData.csv"); // Add text for unique name : Path/InvariantData_	
			if(!writeInvariantFile(modelInvariantFile,modelDistInvariants,modelAngleInvariants,modelPoints.size())	){
				LOG4CPP_ERROR(logger,"Invariant File Writing Error ");
			}
		}
	}
	// Reading marker settings for detection 
	CIRCULAR_MARKER_SETTINGS tempSettings; 
	bool settingsCheck; 
	std::string settingsFilePath = "Settings";
	CIRCULAR_MARKER_SETTINGS markerSettings = Ubitrack::Vision::CircularMarkerDefaults::defaultCircularMarkerSettings; 

	Mat img; 
	// Iterate over images 
	while ( imageIndex < noOfFiles ||  useCamera ) {




		if(useCamera) {

			Mat frame = cvQueryFrame(capture);
			if ( frame.empty() ) {
				LOG4CPP_ERROR(logger,"Camera Frame was not captured ");
				getchar();
				break;
			}
			
			img = frame.clone(); 
			
		}
		else{
			
			if (fileIt == files.end()) 
			{
				fileIt = files.begin();
			}
			img = imread( fileIt->string() );

			// Extract file name from path : "abc.png" from path "D:/xxx/xxx/ddhe/abc.png"
			boost::filesystem3::path temp = fileIt->filename().leaf(); 
			LOG4CPP_INFO(logger, " File Name : " << temp); 

			debugImageName.clear();
			debugImageName = debugPath.data(); 
			// Attach Debug to Path name i.e PATH/debug_
			debugImageName.append("/debug_");

			// attach file name : final name would be PATH/debug_filename.jpg
			temp.replace_extension(".jpg");
			debugImageName.append(temp.string());


			// Snippet to undistrot the image 
#ifdef UNDISTORT_IMAGE
			undistImgName.clear();
			undistImgName = undistImgPath.data();
			undistImgName.append(temp.string());
#endif

			invDataFileName.clear(); 
			invDataFileName = debugPath.data();
			invDataFileName.append("/InvariantData_"); // Add text for unique name : Path/InvariantData_
			// For Invariant data we change extension 
			temp.replace_extension(".csv");	// filename.png -> filename.csv
			invDataFileName.append(temp.string()); // file spacific name : Path/InvarinatData_filenam.csv

		}

		std::vector< Ubitrack::Math::Vector < 2 > > midPoints, projectedPoints;		
		std::vector< int > markerIndexPosition;
		std::vector< bool > markerDetectionStatus;
		std::vector< RotatedRect > ellipses;
		std::vector< std::vector < Point2f >> ellipseContours;
		std::vector<CIRCULAR_MARKER_SETTINGS> detectionResults;

		Ubitrack::Math::Matrix < 3, 3 > intrinsicMatrix, intrinsic; 
		std::vector <std::vector <Point3d> > surfaceNormals; 
		std::vector <std::vector <Point3d> > centerPoints; 
		std::vector<double> projectedRadius;

		double camCalib[9]; 
		// IDS Camera : 12/07/2013	
		// Undistortion Data 
		
		double dist[4]; 
		dist[0]= -0.231714; dist[1]=0.196694; dist[2]=0.000500; dist[3]=0.000067 ;  // IDS:Camera 
		Mat distCoef = Mat(4,1,CV_64F,dist); 
		
		// Calibration used for ellipse manipulation : 
		intrinsicMatrix(0,0) = 3615.516111 ; intrinsicMatrix(1,1) = 3615.663520 ; intrinsicMatrix(2,2) =  1 ;
		intrinsicMatrix(0,2) =  1286.415849;
		intrinsicMatrix(1,2) = 918.451363;
		intrinsicMatrix(0,1) = 0 ; intrinsicMatrix(1,0) = 0 ; intrinsicMatrix(2,0) = 0 ; intrinsicMatrix(2,1) = 0 ; 

		// Web Camera Camera : 29/08/2013	
		// Undistortion Data 
		//double dist[8]; 
		//dist[0]= -2.8964658886681427e-002; dist[1]=1.5969643546717007e+000; dist[2]=9.3322984910516309e-003 ; dist[3]= -8.9803557598603235e-004 ;  
		//dist[4]= 1.7373169396410364e+000; dist[5]=  6.8559427362531690e-003; dist[6]=1.4458907155695153e+000 ; dist[7]=  2.0950517498584000e+000 ;  
		//Mat distCoef = Mat(8,1,CV_64F,dist); 

		//// Calibration used for ellipse manipulation : 
		//intrinsicMatrix(0,0) = 6.6157208286646198e+002 ; intrinsicMatrix(1,1) = 6.6381289761786650e+002 ; intrinsicMatrix(2,2) =  1 ;
		//intrinsicMatrix(0,2) =  3.3651379806376451e+002;
		//intrinsicMatrix(1,2) = 2.4818496341595619e+002;
		//intrinsicMatrix(0,1) = 0 ; intrinsicMatrix(1,0) = 0 ; intrinsicMatrix(2,0) = 0 ; intrinsicMatrix(2,1) = 0 ; 


		// Calibration to be used for Pose Estimation 
		intrinsic = intrinsicMatrix; 

		// Opencv to Ubitrack 
		intrinsic(1,2) = img.cols - 1 - intrinsic(1,2);
		// Convert principal point values from OpenCV (left-handed) to Ubitrack format (right-handed)
		intrinsic(0,2) *= -1;
		intrinsic(1,2) *= -1;
		intrinsic(2,2) *= -1;

		camCalib[0]=intrinsicMatrix(0,0);camCalib[1]=intrinsicMatrix(0,1);camCalib[2]=intrinsicMatrix(0,2);
		camCalib[3]=intrinsicMatrix(1,0);camCalib[4]=intrinsicMatrix(1,1);camCalib[5]=intrinsicMatrix(1,2);
		camCalib[6]=intrinsicMatrix(2,0);camCalib[7]=intrinsicMatrix(2,1);camCalib[8]=intrinsicMatrix(2,2);
		Mat cameraMatrix = Mat(3,3,CV_64F,camCalib);

		if (img.empty()) {
			std::cout << "Error getting image"<<std::endl;
			return -1;
		}

		LOG4CPP_TRACE(logger, " Intrinsic Matrix : \n " <<  intrinsicMatrix );
		LOG4CPP_TRACE(logger, "Calibration Ubi : " << intrinsic); 

#ifdef UNDISTORT_IMAGE
		// Undistorting the image 
		LOG4CPP_INFO(logger, "!!- Undistorting and Saving the image  "  );
		Mat unDist = img.clone();
		undistort(unDist,img,cameraMatrix,distCoef);
		imwrite(undistImgName,img);
#else if
		LOG4CPP_INFO(logger, "!!- No undistortion !!"  );
#endif
		if(useCamera){
			LOG4CPP_INFO(logger, "!!- Using Live Camera ... Undistorting !! "  );
			Mat unDist = img.clone();
			undistort(unDist,img,cameraMatrix,distCoef);
		}
		// Convert to grayscale
		if (img.type()!=CV_8U)
			cvtColor(img, imgGray, CV_RGB2GRAY);
		else 
			imgGray = img.clone();

		// Generate a colored debug image
		cvtColor(imgGray, imgDbg, CV_GRAY2RGB);
		imgTmp = imgGray.clone();

		// Image for final output 
		Mat detectedPointsImage = Mat(img.clone());

		

		double markerDetectionTime = (double) getTickCount(); // for time counting 
		// Update the file as the changes made 
		if(readSettingsFile(settingsFilePath,tempSettings))
		{
			markerSettings = tempSettings; 
		}
		/*markerSettings.imageThresholdAdaptiveWindowSize = 7; 
		markerSettings.imageThresholdIntensityDifference = 5;*/
		findCircularMarkers(imgGray, imgDbg, imgTmp, midPoints, markerIndexPosition, ellipses, ellipseContours, markerSettings, detectionResults, true);
		markerDetectionTime = ( (double)getTickCount() - markerDetectionTime ) / (double)getTickFrequency();  


		// To Calculate a demo pose 
		double residual;
		Math::Pose gtPose; 
		std::vector < Ubitrack::Math::Vector3> worldSurfaceNormals; 
		std::vector<Math::Vector<3,double> >projectedSurfaceNormals; 

		double invariantCalculation = (double) getTickCount(); // for time counting 
		for ( int i= 0 ; i < markerIndexPosition.size() ; i++) {

			std::vector<Point3d> surfaceNorm; 
			std::vector<Point3d> centerPoint; 
			double radius; 

			std::vector<double> paramEllipse(5);
			Ubitrack::Math::Matrix < 3, 3 > conicMatrix; 

			RotatedRect ocvRect ; 
			ocvRect = ellipses.at(markerIndexPosition.at(i)); 

			/////********************** FITTING AND CONVERTING : E3D code***********************
			// This part is just for future use, we might not really have to use it for final application 
			// E3D fitting method for caculating the conic parameters 
			std::vector <Point2f> contourPoints ; 

#ifdef TEST_FITTING

			double majorAxis, minorAxis ; 
			double theta ; 
			double centerX = 1300 ; 
			double centerY = 1300 ; 

			majorAxis = 200;
			minorAxis = 100;
			theta = 90; 
			theta = (theta * CV_PI)/ 180; 
			for (int i=0 ; i < 360 ; i+=10){

				float x = majorAxis * cos(i*CV_PI/180); 
				float y = minorAxis * sin(i*CV_PI/180); 

				float ellipseX = ( cos(theta)*x - sin(theta)*y ) + centerX ; 
				float ellipseY = ( sin(theta)*x + cos(theta)*y ) + centerY ;

				LOG4CPP_TRACE(logger, " Ellipse Points :  " << ellipseX << " : " << ellipseY ) ; 
				contourPoints.push_back(Point2f(ellipseX,ellipseY)); 
			}
			/*		contourPoints.clear();
			contourPoints.push_back(Point2f(1209.1,800.6)); contourPoints.push_back(Point2f(1206.2,803.6));
			contourPoints.push_back(Point2f(1208.5,801.8));	contourPoints.push_back(Point2f(1203.3,804.5));
			contourPoints.push_back(Point2f(1208.0,803.6));	contourPoints.push_back(Point2f(1202.4,805.0));
			contourPoints.push_back(Point2f(1201.5,806.0));	contourPoints.push_back(Point2f(1.1999,805.5));*/
			// Only for Unit Test Data / Otherwise ocvRect is already assigned with current ellipse info 
			ocvRect =  fitEllipse( contourPoints); 

#else if
			// If not testing then we want to deal with contour points directly 
			contourPoints = ellipseContours.at(i); 
#endif
			/* // Snippet to Test the Fitting 

			for(int i = 0 ; i < contourPoints.size(); i++){
			circle( imgDbg, cvPoint(contourPoints.at(i).x,contourPoints.at(i).y), 1 , CV_RGB(255,0,0)); // RED 
			}

			*/ 

			// ########### Conic Matrix Method 1 : E3D Customised ###############

			/* ------- E3D Custom Method Not used for Now -----------Since we do fitting all over again we waste time for now (17/07/2013) 

			// If we just have the points we can use this method to Extract Conic Matrix from Points 
			double residualError; 
			residualError = fitEllipseE3D( contourPoints, paramEllipse); 

			// Get conic matrix from the conic parameters to be used for functionality 
			getConicMatrix( paramEllipse, conicMatrix); 
			LOG4CPP_DEBUG( logger, " Conic matrix by E3D Ellipse Fitting :  \n" << conicMatrix); 

			// -> Verification of algorithm 
			// converting conic parameters to Rect , to verify our algorithm with OpenCV
			RotatedRect tempRect;
			convertConic2Rect(paramEllipse, tempRect) ;

			// Drawing Fitting Results from both E3D : As overlay on the contour points to check fitting quality 
			ellipse( imgDbg , tempRect, CV_RGB(0,0,255), 1 );  

			// ----End Method 1 
			----------------- E3D custom method end ---- To be used during final testing for accuracy measure */ 

			// ########### Conic Matrix Method 2 : Ocv Parameters to E3D using converter ################
			// What we understand about openCV?? FItting algorithm 
			// Opencv starts the drawing ellipse from major axis , at angle it makes with the x axis  

			// But the major axis is mapped to width while fitting ellipse and width is forced to be smaller than height 
			// So the angle is always 90 more than actual 


			// Converting conic parameters from Ocv data : Scaled (i.e. F forced to 1)
			convertRect2Conic( ocvRect , paramEllipse) ;		

			// Get conic matrix from the conic parameters to be used for functionality 
			getConicMatrix( paramEllipse, conicMatrix); 
			LOG4CPP_DEBUG( logger, "conic equation by Ocv Fitting and E3D parameter converter :  \n" << conicMatrix); 

			// Drawing Fitting Results from both Ocv  : To check fitting quality by overlaying ellipse on contour points 
			// ellipse( imgDbg , ocvRect , CV_RGB(255,0,255), 1 ); 

			//////////*****************End Method 2**************************////////////////////

			double markerWorldRadius = markerSize.at(0)/2 ; // in mili-meters 
			LOG4CPP_DEBUG( logger, " Radius Considered  :  \n" << markerWorldRadius); 

			getSurfaceNormals( ellipseContours.at(i), conicMatrix, intrinsicMatrix, surfaceNorm, centerPoint, radius, markerWorldRadius );

			surfaceNormals.push_back(surfaceNorm);
			projectedRadius.push_back(radius);
			centerPoints.push_back(centerPoint); 

		}
		invariantCalculation = ( (double)getTickCount() - invariantCalculation ) / (double)getTickFrequency();  

		std::vector< std::vector <double> > calculatedDistInvariants; 
		std::vector< std::vector <double> > calculatedAngleInvariants; 
		std::vector <std::vector <int> > calculatedPointPair; 
		std::vector<std::vector<int>>  indexMatching; 
		multimap<int ,int > matchingMap, updatedMatchingMap; 
		int distThresh = 10; // 10 // in mm 
		int angleThresh = 5 ; // 5 // in Degree
		std::vector<std::vector<int>>  modelImageMatchedPair; 
		std::vector<std::vector<Math::Vector3>>  matchingTriplet; 
		Mat votingMatrix = Mat::zeros(midPoints.size(),modelPoints.size(),CV_64F);
		Mat weighingMatrix = Mat::zeros(midPoints.size(),modelPoints.size(),CV_64F);

		double matchingCalculation = (double) getTickCount();
		if( midPoints.size() != 0) {

			getPairCombinations(midPoints.size(),calculatedPointPair); 

			if(! calculateConicDistanceInvariants(centerPoints, calculatedDistInvariants)){
				LOG4CPP_ERROR(logger,"Calculating Invariants Failed");
			} 
			if( !calculateConicAngleInvariants(surfaceNormals,calculatedAngleInvariants)) {
				LOG4CPP_ERROR(logger,"Calculating Invariants Failed");
			} 
			if(DEBUG_INVARIANT_DATA){
				if(!writeInvariantFile(invDataFileName,calculatedDistInvariants,calculatedAngleInvariants,centerPoints.size())	){
					LOG4CPP_ERROR(logger,"Invariant File Writing Error ");
				}
			}

			// Find matching based on invariants , Hypothesis for matching pairs 
			findMatchingPointPairs(calculatedDistInvariants,calculatedAngleInvariants,modelDistInvariants,modelAngleInvariants,indexMatching,distThresh,angleThresh);

			// Filter the matching for final 2D-3D match result 
			// From given triplets finding the correspondence map 


			filterPointPair(indexMatching,calculatedPointPair,modelPointPair, modelImageMatchedPair, matchingMap, matchingTriplet, votingMatrix); 
			LOG4CPP_INFO(logger, "Matching Mat \n " << votingMatrix ); 

			// From given triplets finding the correspondence map 
			// 27-08-13 Removing triplet info passing as its irrelevant 
			createCorrespondenceMap(matchingMap, votingMatrix , weighingMatrix); 

		}
		else{

			LOG4CPP_DEBUG(logger, "No markers found "); 

		}

		matchingCalculation = ( (double)getTickCount() - matchingCalculation ) / (double)getTickFrequency();  

		// Pose Detection 
		std::vector<Math::Vector<3>> model3DPoint, allModelPoint; 
		std::vector<Math::Vector3> Image3DPoint; 
		std::vector<Math::Vector<2>> imagePoints; 

		for(int i = 0; i < midPoints.size() ; i++){

			if( matchingMap.find(i) != matchingMap.end() ){
				int mappedModelPoint = matchingMap.find(i)->second; 			
				Image3DPoint.push_back(Math::Vector3(-centerPoints.at(i).at(0).x,centerPoints.at(i).at(0).y,centerPoints.at(i).at(0).z) );
				model3DPoint.push_back(Math::Vector3(modelPoints.at(mappedModelPoint).at(0).x,modelPoints.at(mappedModelPoint).at(0).y,modelPoints.at(mappedModelPoint).at(0).z));
				imagePoints.push_back(midPoints.at(i));
			}

		}

		for(int i = 0 ; i < modelPoints.size(); i++) {
			allModelPoint.push_back(Math::Vector3(modelPoints.at(i).at(0).x,modelPoints.at(i).at(0).y,modelPoints.at(i).at(0).z));
		}


		double poseEstimationTime = (double)getTickCount();

		// NOTE: For now we skip doing it the hard way. 
		//if( model3DPoint.size() > 2) {
		//	
		//	try{
		//	// For pose with 3 points 
		//	Math::Pose poseAbsoluteOrentation = Calibration::calculateAbsoluteOrientation(  model3DPoint, Image3DPoint );
		//	double rms = Calibration::computeRms( poseAbsoluteOrentation , model3DPoint,  Image3DPoint );
		//	drawPoseE3D( detectedPointsImage, poseAbsoluteOrentation , intrinsic, 3, rms, 0, 0.008 );
		//	LOG4CPP_INFO(logger,"Pose Absolute Orientation : " << poseAbsoluteOrentation); 
		//	}
		//	catch(std::exception&){
		//		LOG4CPP_ERROR(logger, " Pose computation Error in absolute oritentation " );
		//	}
		//}

		double rms2 = 0 ;	
		Math::ErrorPose pose2; 
		bool gotPose = false; 

		if(imagePoints.size() > 4 ) {


			try {

				pose2 = Ubitrack::Calibration::computePose(imagePoints,model3DPoint,intrinsic, rms2, true, Calibration::NONPLANAR_PROJECTION);
				LOG4CPP_INFO(logger," 2D3D Pose Estimation : "<<  pose2 << ": Error : " << rms2 ); 
				// drawPoseE3D( detectedPointsImage, pose2, intrinsic, 3, rms2, 0, 0.008 );
				gotPose = true ;

			} catch (std::exception& ) { // Ubitrack::Util::Exception& e
				LOG4CPP_ERROR(logger, " Pose computation Error in 2D-3D pose estimation  " );
			}

		}

		else{

			LOG4CPP_INFO(logger," Sorry no pose with No 6 point matching :-(   "); 
		}

		// Only if Pose was calculated in the first place 
		if(gotPose){
			Math::Matrix < 3, 4 > projectionMatrix = intrinsic * pose2;  
			updatedMatchingMap = matchingMap; // Let the map be in Matching already 
			match2d3dpoints(midPoints,allModelPoint,projectionMatrix, updatedMatchingMap, 15.0); 

			for(multimap<int,int>::iterator it = updatedMatchingMap.begin(); it != updatedMatchingMap.end() ; it++){			
				model3DPoint.push_back(Math::Vector3(modelPoints.at((*it).second).at(0).x,modelPoints.at((*it).second).at(0).y,modelPoints.at((*it).second).at(0).z)); 
				imagePoints.push_back(midPoints.at((*it).first)); 
				weighingMatrix.at<double>((*it).first,(*it).second) = weighingMatrix.at<double>((*it).first,(*it).second)+1 ; 
			}

			try {

				pose2 = Ubitrack::Calibration::computePose(imagePoints,model3DPoint,intrinsic, rms2, true, Calibration::NONPLANAR_PROJECTION);
				LOG4CPP_INFO(logger," Refined 2D3D Pose : "<<  pose2 << ": Refined Error : " << rms2 ); 
				gotPose = true ;

			} catch (std::exception& ) { // Ubitrack::Util::Exception& e
				LOG4CPP_ERROR(logger, " Refined Pose calculation failed " );
			}

		}

		// Draw Pose 
		if(gotPose)
			drawPoseE3D( detectedPointsImage, pose2, intrinsic, 3, rms2, 0, 0.008 );


		LOG4CPP_INFO(logger," Weighted Matrix \n " << weighingMatrix );
		// Flip to plot back into the image 
		flipY(midPoints, IplImage(imgDbg).height );
		poseEstimationTime = ( (double)getTickCount() - poseEstimationTime ) / (double)getTickFrequency(); 



		std::cout << "Found "<< midPoints.size() << " markers.\n";
		std::cout << "Time Taken:  "<< markerDetectionTime << " Secs.\n";
		std::cout << "Time Taken for invariant calculation :  "<< invariantCalculation << " Secs .\n";
		std::cout << "Time Taken for matching points :  "<< matchingCalculation << " Secs .\n";
		std::cout << "Time Taken for pose estimation:  "<< poseEstimationTime << " Secs .\n";

		for(int i = 0 ; i < markerIndexPosition.size(); i++){

			RotatedRect markerRect = ellipses.at(markerIndexPosition.at(i)); 
			string message;
			message.append(boost::lexical_cast<string> (i) );

			cv::Scalar colorMatched = CV_RGB(0,255,0);
			cv::Scalar colorNonMatched = CV_RGB(255,0,0);
			cv::Scalar colorText = CV_RGB(235,0,250);
			cv::Scalar colorUpdatedPoints = CV_RGB(0,0,250);
			circle( detectedPointsImage , cvPoint(markerRect.center.x,markerRect.center.y), 5, colorNonMatched, 4 );

			if( updatedMatchingMap.find(i) != updatedMatchingMap.end()) {
				if(!useCamera){
					int modelPointIndex = updatedMatchingMap.find(i)->second; 
					message.append("#");
					message.append(boost::lexical_cast<string>( modelPointIndex ) ) ;  // modelpoint index 
					message.append("#");
					message.append(boost::lexical_cast<string>( modelPointIDs.at( modelPointIndex ) ) ) ;  // prints mapped original ID 
					message.append("#");
					message.append(boost::lexical_cast<string>( votingMatrix.at<double>(i,modelPointIndex) ) ) ;  // vote for the point  
				}
				circle( detectedPointsImage , cvPoint(markerRect.center.x,markerRect.center.y), 5, colorUpdatedPoints, 4 );
			}

			std::stringstream ss;
			ss << i;
			int fontFace = FONT_HERSHEY_DUPLEX;
			double fontScale = 1;
			if(img.cols < 1000 && img.rows < 1000)
				fontScale = 0.8;

			std::pair <std::multimap<int,int>::iterator, std::multimap<int,int>::iterator> ret;
			ret = matchingMap.equal_range(i);

			for (std::multimap<int,int>::iterator it=ret.first; it!=ret.second; ++it){
				if(!useCamera){
					message.append("#");
					message.append(boost::lexical_cast<string>( (it->second) ) ) ;  // modelpoint index 
					message.append("#");
					message.append(boost::lexical_cast<string>( modelPointIDs.at( (it->second) ) ) ) ;  // prints mapped original ID 
					message.append("#");
					message.append(boost::lexical_cast<string>( votingMatrix.at<double>(i,it->second) ) ) ;  // vote for the point  
				}
				circle( detectedPointsImage , cvPoint(markerRect.center.x,markerRect.center.y), 5, colorMatched, 4 );
			}

			putText( detectedPointsImage, message , cvPoint(markerRect.center.x+20,markerRect.center.y+20) ,fontFace, fontScale, colorText );

		}
		if(useCamera){
			imshow("Tracker",detectedPointsImage);
		}
		else{
			imwrite(debugImageName,detectedPointsImage); // save image 
		}

		// Show onl if asked 
		if(  show_image_active){

			namedWindow("Final Image", 0);
			imshow("Final Image",detectedPointsImage);

		}

		if(show_debug_window ) { 

			namedWindow("Bin", CV_GUI_EXPANDED);
			imshow("Bin",imgTmp);
			setMouseCallback( "Bin", my_mouse_callback);

			namedWindow("Debug", CV_GUI_EXPANDED);
			imshow("Debug", imgDbg);
			setMouseCallback( "Debug", my_mouse_callback);

			showDebugWindows(imgGray, imgTmp, imgDbg, "Mouse ", 200, 200, mx, my);
		}
		
		if(useCamera){
			if ( (cvWaitKey(10) & 255) == 27 ) break;
		}
		else{
			int key = waitKey(0);

			// Give control when either debug or only final image show is on 
			if(show_debug_window || show_image_active){


				if (key==(int)('+')) {
					markerSettings.imageThresholdIntensityDifference+=1.0;
					std::cout << markerSettings.imageThresholdIntensityDifference << "\n";
				}
				if (key==(int)('-')) {
					markerSettings.imageThresholdIntensityDifference-=1.0;
					std::cout << markerSettings.imageThresholdIntensityDifference << "\n";
				}

				if (key==(int)('x')) {
					if (!useCamera)
						fileIt++;
					imageIndex++;
				}

				if (key==(int)('y')) {
					if (!useCamera)
						fileIt--;
					imageIndex--;
				}

				if (key==(int)(' ')) {
					detect = !detect;
				}

			}
			// Automatic increment if everything is off 
			if(!show_debug_window && !show_image_active){
				fileIt++;
				imageIndex++;
			}
			if (key==27)
				break;
		}
	}
	if(useCamera){
		cvReleaseCapture( &capture );
		cvDestroyWindow( "Tracker" );
		
	}
	return 0; 

}
