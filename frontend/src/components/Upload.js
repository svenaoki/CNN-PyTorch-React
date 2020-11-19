import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Upload.css";

const Upload = () => {
  const [file, setFile] = useState("");
  const [predDog, setPredDog] = useState("");
  const [predCat, setPredCat] = useState("");
  const [img, setImg] = useState(
    "https://storage.googleapis.com/petbacker/images/blog/2018/cat-vs-dog.jpg"
  );
  const [loading, setLoading] = useState(true);

  const handleImageChange = (e) => {
    setPredCat("");
    setPredDog("");
    let reader = new FileReader();
    let file = e.target.files[0];
    reader.onloadend = () => {
      setImg(reader.result);
      setFile(file);
    };
    reader.readAsDataURL(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    let form_data = new FormData();
    form_data.append("file", file);

    let url = "http://localhost:5000";
    axios
      .post(url, form_data, {
        headers: {
          "content-type": "multipart/form-data",
        },
      })
      .then((res) => {
        setPredDog(res.data[1]);
        setPredCat(res.data[0]);
        setLoading(false);
      })
      .catch((err) => console.log(err));
  };

  const loadingSpinner = (
    <div>
      <div className="spinner-grow text-secondary" role="status">
        <span className="sr-only">Loading...</span>
      </div>
      <div className="spinner-grow text-secondary" role="status">
        <span className="sr-only">Loading...</span>
      </div>
      <div className="spinner-grow text-secondary" role="status">
        <span className="sr-only">Loading...</span>
      </div>
    </div>
  );

  const results = (
    <div>
      <p>P(dog|data): {predDog * 100}% </p>
      <p>P(cat|data): {predCat * 100}% </p>
    </div>
  );

  return (
    <div className="container">
      <div className="panel panel-primary">
        <div className="panel-heading">
          <h3 className="panel-title-primary">
            Predicting cat/dog using neural nets
          </h3>
        </div>
        <div className="panel-body">
          <img src={img} className="figure-img img-fluid rounded" />
          <h4>{loading ? loadingSpinner : results}</h4>
          <form onSubmit={handleSubmit}>
            <input
              type="file"
              id="image"
              accept="image/png, image/jpeg"
              onChange={handleImageChange}
              required
            />
            <hr />
            <input className="btn btn-primary" type="submit" value="Run CNN" />
          </form>
        </div>
      </div>
    </div>
  );
};

export default Upload;
