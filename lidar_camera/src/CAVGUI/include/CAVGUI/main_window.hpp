/**
 * @file /include/CAVGUI/main_window.hpp
 *
 * @brief Qt based gui for CAVGUI.
 *
 * @date November 2010
 **/
#ifndef CAVGUI_MAIN_WINDOW_H
#define CAVGUI_MAIN_WINDOW_H

/*****************************************************************************
** Includes
*****************************************************************************/

#include <QtGui/QMainWindow>
#include "ui_main_window.h"
#include "qnode.hpp"
#include <QTimer>
#include <QProcess>

/*****************************************************************************
** Namespace
*****************************************************************************/

namespace CAVGUI {

/*****************************************************************************
** Interface [MainWindow]
*****************************************************************************/
/**
 * @brief Qt central, all operations relating to the view part here.
 */
class MainWindow : public QMainWindow {
Q_OBJECT

public:
	MainWindow(int argc, char** argv, QWidget *parent = 0);
	~MainWindow();

	void ReadSettings(); // Load up qt program settings at startup
	void WriteSettings(); // Save qt program settings when closing

	void closeEvent(QCloseEvent *event); // Overloaded function
	void showNoMasterMessage();

public Q_SLOTS:
	/******************************************
	** Auto-connections (connectSlotsByName())
	*******************************************/
	void on_actionAbout_triggered();
	void on_button_connect_clicked(bool check );
	void on_checkbox_use_environment_stateChanged(int state);

	//add by wqyang@umich.edu
	void on_button_rosbag_clicked();
	void on_button_rviz_clicked();

    /******************************************
    ** Manual connections
    *******************************************/
    void updateLoggingView(); 
	//add by wqyang@umich.edu
	void rosbagRecordUpdate();

private:
	Ui::MainWindowDesign ui;
	QNode qnode;
	QTimer pTimer;
	QProcess rosbagProcess;
	
};

}  // namespace CAVGUI

#endif // CAVGUI_MAIN_WINDOW_H
