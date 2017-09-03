import React, { Component } from 'react';
import './App.css';
import './css/form-elements.css';
import Navbar from './Navbar';
import StepForm from './StepForm';

class App extends Component {
  render() {
    return (
      <div className="App">
        <Navbar />
        
        <div className="top-content App-content">

        <div className="inner-bg">
                <div className="container">
                    <div className="row">
                        <div className="col-sm-8 col-sm-offset-2 text">
                            <h1><strong>Diabetes</strong> detection & prediction.</h1>
                            <div className="description">
                              <p>
                                Enter your data and we&#39;ll analyze your chances of being diabetic.
                              </p>
                            </div>
                        </div>
                    </div>
                    <div className="row">
                        <div className="col-sm-6 col-sm-offset-3 form-box">
                          
                          <StepForm />
                        
                        </div>
                    </div>
                </div>
              </div>
        </div>
      </div>
    );
  }
}

export default App;
