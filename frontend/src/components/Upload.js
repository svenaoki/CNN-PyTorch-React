import React, { useState, useEffect } from "react";
import axios from "axios";
import "./Upload.css";

const Upload = () => {
  const [file, setFile] = useState("");
  const [pred, setPred] = useState("Make a prediction");
  const [img, setImg] = useState(
    "https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_960_720.png"
  );
  const [loading, setLoading] = useState(true);

  const handleImageChange = (e) => {
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
        setPred(res.data["class_id"] === "0" ? "Dog" : "Cat");
        setLoading(false);
      })
      .catch((err) => console.log(err));
  };

  const loadingSpinner = (
    <div className="spinner-grow text-secondary" role="status">
      <span className="sr-only">Loading...</span>
    </div>
  );

  return (
    <div className="container">
      <img src={img} className="figure-img img-fluid rounded" />
      <br />
      <h3>{loading ? loadingSpinner : pred}</h3>
      <br />
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
  );
};

export default Upload;
