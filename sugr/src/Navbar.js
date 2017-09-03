import React, { Component } from 'react';

class Navbar extends Component {
	render() {
		return (
			<nav className="navbar navbar-inverse navbar-no-bg App-header" role="navigation">
	          <div className="container">
	            <div className="navbar-header">
	              <button type="button" className="navbar-toggle collapsed" data-toggle="collapse" data-target="#top-navbar-1">
	                <span className="sr-only">Toggle navigation</span>
	                <span className="icon-bar"></span>
	                <span className="icon-bar"></span>
	                <span className="icon-bar"></span>
	              </button>
	              <a className="navbar-brand" href="index.html">Sugr</a>
	            </div>
	            <div className="collapse navbar-collapse" id="top-navbar-1">
	              <ul className="nav navbar-nav navbar-right">
	                <li>
	                  <span className="li-social">
	                    <a href="#"><i className="fa fa-facebook"></i></a> 
	                    <a href="#"><i className="fa fa-twitter"></i></a> 
	                    <a href="#"><i className="fa fa-envelope"></i></a> 
	                    <a href="#"><i className="fa fa-skype"></i></a>
	                  </span>
	                </li>
	              </ul>
	            </div>
	          </div>
	        </nav>
	    )
	}
};

export default Navbar;