<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<!-- [![Contributors][contributors-shield]][contributors-url]
[![MIT License][license-shield]][license-url] -->


<!-- PROJECT LOGO -->
<br />
<!-- <div align="center">
  <a href="https://github.com/github_username/repo_name">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a> -->

<h3 align="center">Federated Generalized Linear Mixed model Association Tests (FedGMMAT)</h3>

  <p align="center">
    FedGMMAT is developed in Python based on R package <a href="https://github.com/hanchenphd/GMMAT">GMMAT</a> by Dr. Chen Han
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#prerequisites">Prerequisites</a></li>
      </ul>
    </li>
<!--     <li><a href="#usage">Usage</a></li> -->
<!--     <li><a href="#roadmap">Roadmap</a></li> -->
<!--     <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#citations">Citations</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

__Welcome__ to the FedGMMAT GitHub repository!

FedGMMAT is a privacy-preserving genome-wide association study (GWAS) algorithm that utilizes federated learning to protect the privacy of individuals' genetic data. This algorithm is based on the R package GMMAT, developed by Dr. Chen Han, and extends it to support federated learning.

GWAS aims to identifying genetic variants that are associated with diseases and traits. However, the use of GWAS data raises concerns about privacy and confidentiality, as genetic data can be used to identify individuals. FedGMMAT addresses these concerns by enabling multiple parties to collaboratively analyze their GWAS data while keeping it private.

We hope that FedGMMAT will be useful for researchers and practitioners who are interested in conducting privacy-preserving GWAS analyses. Please feel free to explore the repository, use the software, and contribute to its development.
<!-- Here's a blank template to get started: To avoid retyping too much info. Do a search and replace with your text editor for the following: `github_username`, `repo_name`, `twitter_handle`, `linkedin_username`, `email_client`, `email`, `project_title`, `project_description` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [Python](Python-url)
* [PyTorch](PyTorch-url)
* [Pandas](Pandas-url)
* [SciPy](SciPy-url)
* [TenSEAL](TenSEAL-url)
<!-- * [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url] -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Please have conda or anaconda ready in your environment.

### Installation

Clone this repository to your local device.
  ```sh
  git clone https://github.com/Li-Wentao/FedGMMAT.git
  ```

### Prerequisites

Install the dependency of the package environment.
* Conda
  ```sh
  conda create --name FedGMMAT --file requirements.txt
  ```

<!-- ### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ``` -->

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Quick start

Please refer to the `FedGMMAT tutorial.ipynb` file. Example data is provided in directory `example_data`, the data are pulled from example data from R package _GMMAT_.

<!-- USAGE EXAMPLES -->
<!-- ## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>
 -->


<!-- ROADMAP -->
<!-- ## Roadmap

- [ ] Feature 1
- [ ] Feature 2
- [ ] Feature 3
    - [ ] Nested Feature

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- CONTRIBUTING -->
<!-- ## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p> -->



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Wentao Li - [email](wentao.li@uth.tmc.edu)

Project Link: [https://github.com/Li-Wentao/FedGMMAT.git](https://github.com/Li-Wentao/FedGMMAT.git)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Dr. Han Chen](https://sbmi.uth.edu/faculty-and-staff/han-chen.htm)
* [Dr. Arif Harmanci](https://sbmi.uth.edu/faculty-and-staff/arif-harmanci.htm)
* [Dr. Xiaoqian Jiang](https://sbmi.uth.edu/faculty-and-staff/xiaoqian-jiang.htm)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- Publications -->

## Citations

> Please cite the following publications

Chen, H., Wang, C., Conomos, M.P., Stilp, A.M., Li, Z., Sofer, T., Szpiro, A.A., Chen, W., Brehm, J.M., Celedón, J.C. and Redline, S., ["Control for Population Structure and Relatedness for Binary Traits in Genetic Association Studies via Logistic Mixed Models"](https://pubmed.ncbi.nlm.nih.gov/27018471/), 98(4), pp.653-666.

```
@article{chen2016control,
  title={Control for population structure and relatedness for binary traits in genetic association studies via logistic mixed models},
  author={Chen, Han and Wang, Chaolong and Conomos, Matthew P and Stilp, Adrienne M and Li, Zilin and Sofer, Tamar and Szpiro, Adam A and Chen, Wei and Brehm, John M and Celed{\'o}n, Juan C and others},
  journal={The American Journal of Human Genetics},
  volume={98},
  number={4},
  pages={653--666},
  year={2016},
  publisher={Elsevier}
}
```

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!-- [contributors-shield]: https://img.shields.io/github/contributors/github_username/repo_name.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo_name/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo_name.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo_name/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo_name.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo_name/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo_name.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo_name/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo_name.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo_name/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/linkedin_username
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com  -->
[Python]: https://www.python.org/static/img/python-logo.png
[Python-url]: https://www.python.org/
[PyTorch]: https://github.com/pytorch/pytorch/raw/master/docs/source/_static/img/pytorch-logo-dark.png
[PyTorch-url]: https://pytorch.org/
[Pandas]: https://pandas.pydata.org/static/img/pandas_white.svg
[Pandas-url]: https://pandas.pydata.org/
[SciPy]: https://scipy.org/images/logo.svg
[SciPy-url]: https://scipy.org/
[TenSEAL-url]: https://github.com/OpenMined/TenSEAL.git