import React, { Component } from 'react';
import './css/form-elements.css';

class StepForm extends Component {
	constructor(props) {
		super(props);
		this.state = {
			female: false
		};
	}
	handleGenderChange(e) {

		this.setState({
			gender: e.target.value
		});
	}
	render() {
		return (

			<form role="form" action="" method="post" className="registration-form">
                            
                <fieldset>
                  <div className="form-top">
                    <div className="form-top-left">
                      <h3>Step 1 / 2</h3>
                        <p>Tell us who you are:</p>
                    </div>
                    <div className="form-top-right">
                      <i className="fa fa-user"></i>
                    </div>
                    </div>
                    <div className="form-bottom">
                  <div className="form-group">
                    <label className="sr-only" htmlFor="form-first-name">Name</label>
                      <input type="text" name="form-first-name" placeholder="ENTER NAME" className="form-first-name form-control" id="form-first-name" />
                    </div>
                    <div className="form-group">
                      <label className="sr-only" htmlFor="form-last-name">Email</label>
                      <input type="email" name="form-last-name" placeholder="ENTER EMAIL" className="form-last-name form-control" id="form-last-name" />
                    </div>
                    <div className="form-group">
                      <label className="sr-only" htmlFor="form-last-name">Gender</label>
                      <select name="form-gender" id="form-gender" onChange={this.handleGenderChange.bind(this)}>
                        <option value="">ENTER GENDER</option>
                        <option value="m">MALE</option>
                        <option value="f">FEMALE</option>
                      </select>
                    </div>
                    <div className="form-group">
                      <label className="sr-only" htmlFor="form-last-name">Age</label>
                      <input type="text" name="form-last-name" placeholder="ENTER AGE" className="form-last-name form-control" id="form-last-name" />
                    </div>
                    <button type="button" className="btn btn-next">Next</button>
                </div>
              </fieldset>
              
              <fieldset>
                  <div className="form-top">
                    <div className="form-top-left">
                      <h3>Step 2 / 2</h3>
                        <p>Enter your data:</p>
                    </div>
                    <div className="form-top-right">
                      <i className="fa fa-database"></i>
                    </div>
                    </div>
                    <div className="form-bottom">
                    <div className="form-group">
                      <label className="sr-only" htmlFor="form-email">BMI</label>
                      <input type="text" name="form-bmi" placeholder="BMI" className="form-bmi form-control" id="form-bmi" />
                    </div>
                    <div className="form-group">
                    <label className="sr-only" htmlFor="form-password">Blood Pressure</label>
                      <input type="text" name="form-bp" placeholder="BLOOD PRESSURE (DIASTOTIC)" className="form-bp form-control" id="form-bp" />
                    </div>

                    {this.state.gender==="f"?
                    <div className="form-group">
                    <label className="sr-only" htmlFor="form-password">No. of Pregnancies</label>
                      <input type="number" name="form-bp" placeholder="NUMBER OF PREGNANCIES" className="form-preg form-control" id="form-preg" />
                    </div>
                    : null}
                    <button type="button" className="btn btn-previous">Previous</button>
                    &nbsp;
                    <button type="button" className="btn btn-next">Next</button>
                </div>
              </fieldset>
              
              <fieldset>
                  <div className="form-top">
                    <div className="form-top-left">
                      <h3>Result</h3>
                        <p>You:</p>
                    </div>
                    <div className="form-top-right">
                      <i className="fa fa-line-chart"></i>
                    </div>
                    </div>
                    <div className="form-bottom">
                
                  </div>
              </fieldset>
            
            </form>

		)
	}
};

export default StepForm;