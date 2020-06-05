#include "ukf.h"
#include "Eigen/Dense"
#include "iostream"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1,  0,  0,  0,  0,
        0,  1,  0,  0,  0,
        0,  0,  1,  0,  0,
        0,  0,  0,  1,  0,
        0,  0,  0,  0,  1;
  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  // initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // State dimension [pos1 pos2 vel_abs yaw_angle yaw_rate]
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;
  
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);
  Xsig_pred_.fill(0.0);

  // time when the state is true, in us
  time_us_ = 0.0;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);
  double weight_0 = lambda_/(lambda_+n_aug_);
  double weight = 0.5/(lambda_+n_aug_);
  weights_(0) = weight_0;

  for (int i=1; i<2*n_aug_+1; ++i) 
  {  
    weights_(i) = weight;
  }

}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_)
  {
    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
        // Initialize the state with the initial location and zero velocity, yaw and yaw rate
        double px = meas_package.raw_measurements_[0];
        double py = meas_package.raw_measurements_[1];
        x_ << px,
              py,
              0,
              0,
              0;
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
        double rho = meas_package.raw_measurements_[0];
        double phi = meas_package.raw_measurements_[1];
        double rho_dot = meas_package.raw_measurements_[2];

        double px = rho*cos(phi);
        double py = rho*sin(phi);
        double vx = rho_dot*cos(phi);
        double vy = rho_dot*sin(phi);
        double v = sqrt(vx*vx + vy*vy);
        x_ << px,
              py,
              v,
              0,
              0;
    }
    else
    {
      std::cout << "Invalid Sensor Type" << std::endl;
    }

    time_us_ = meas_package.timestamp_; // set the previous time step with the current measurement timestamp
    is_initialized_ = true;
    return;
    
  }

  // compute the time elapsed between the current and the previous measurements
  // dt - expressed in seconds
  double dt = (meas_package.timestamp_ - time_us_) / 1.0e6;
  time_us_ = meas_package.timestamp_;

  // Prediction Step
  Prediction(dt);

  // Measurement Update Step
  if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
  }
  else
  {
    std::cout << "Invalid Sensor Type" << std::endl;
  }
  
  std::cout << "State Vector x_ = " << x_ << std::endl;

}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  /* ------------------------ Generate and Augment Sigma points -------------------------- */

  // create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
 
  // create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; ++i) 
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }

/* ------------------------ Sigma points prediction -------------------------- */
  // predict sigma points
  for (int i = 0; i< 2*n_aug_+1; ++i) 
  {
    double p_x   = Xsig_aug(0,i);
    double p_y   = Xsig_aug(1,i);
    double v     = Xsig_aug(2,i);
    double yaw   = Xsig_aug(3,i);
    double yawd  = Xsig_aug(4,i);
    double nu_a  = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    // predicted state values
    double px_p, py_p;

    // avoid division by zero
    if (fabs(yawd) > 0.001) 
    {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else 
    {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

/* ------------------------ Mean and Covariance Prediction -------------------------- */
  
  // Predicted state vector
  VectorXd x_p = VectorXd(n_x_);
  x_p.fill(0.0);
  
  // iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
      x_p = x_p + weights_(i) * Xsig_pred_.col(i);
  }

  // Predicted covariance matrix
  MatrixXd P_p = MatrixXd(n_x_,n_x_);
  P_p.fill(0.0);

  // iterate over sigma points
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
    
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_p;
    // angle normalization
    NormalizeAngle(x_diff,3);

    P_p = P_p + weights_(i) * x_diff * x_diff.transpose() ;
  }

  // Modify the state vector and state covariance matrix with the predicted result
  x_ = x_p;
  P_ = P_p;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
    /* ------------------------ Lidar Measurement Prediction -------------------------- */

  // Lidar measurement dimensions [x y]
  int lidar_meas_dimension = 2;

  // Innovation Covariance Matrix
  MatrixXd S;

  // Kalman gain K;
  MatrixXd K;

  // residual
  VectorXd z_diff;

  UkfUpdate(meas_package, lidar_meas_dimension, &S, &K, &z_diff);

  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  // Radar measurement dimensions [rho phi rho_dot]
  int radar_meas_dimension = 3;

  // Innovation Covariance Matrix
  MatrixXd S;
  // Kalman gain K;
  MatrixXd K;

  // residual
  VectorXd z_diff;

  UkfUpdate(meas_package, radar_meas_dimension, &S, &K, &z_diff);
  
  // update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

}

/* Common UKF update function that output the Innovation Covariance Matrix (S), the Kalman Gain (K) and the Residual between actual and predicted values
   for each sensor */
void UKF::UkfUpdate(MeasurementPackage meas_package, int n_z, MatrixXd* S_out, MatrixXd* K_out, VectorXd* z_diff_out)
{
  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v   = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      // measurement model
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                       // rho
      Zsig(1,i) = atan2(p_y,p_x);                                // phi
      Zsig(2,i) = (p_x*v1 + p_y*v2) / sqrt(p_x*p_x + p_y*p_y);   // rho_dot
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      Zsig(0,i) = p_x;                       // x
      Zsig(1,i) = p_y;                       // y
    }
    
  }

  // mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2*n_aug_+1; ++i) 
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  // innovation covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
  {  
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    NormalizeAngle(z_diff,1);

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_z,n_z);
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    R <<  std_radr_*std_radr_,            0,            0,
                0,           std_radphi_*std_radphi_, 0,
                0,                      0,            std_radrd_*std_radrd_;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    R <<  std_laspx_*std_laspx_,         0,           
                  0,         std_laspy_*std_laspy_;
  }
  
  S = S + R;


  // Incoming Radar Measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0.0);

  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // 2n+1 simga points
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    NormalizeAngle(z_diff,1);

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    NormalizeAngle(x_diff,3);

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  NormalizeAngle(z_diff,1);

  *K_out = K;
  *S_out = S;
  *z_diff_out = z_diff;
}

  void UKF::NormalizeAngle(Eigen::VectorXd & vector, int index)
  {
      while (vector(index)> M_PI) vector(index)-=2.*M_PI;
      while (vector(index)<-M_PI) vector(index)+=2.*M_PI;
  }
