:html_theme.sidebar_secondary.remove: true

ModelAtlas
==========

.. toctree::
   :maxdepth: 2
   :hidden:

   Start here <start/index>
   Examples <examples>
   User guide <guide/index>
   API reference <api>
   Development <development/index>

.. raw:: html

   <div class="atlas-home">
     <div class="atlas-hero">
       <div class="atlas-hero-copy">
         <p class="atlas-kicker">Scientific Python <span aria-hidden="true">/</span> Research preview</p>
         <h1>Visualize how astronomical models learn</h1>
         <p class="atlas-lede">Explore SOMs, neural networks, and statistical models on real galaxy, stellar, and solar observations. Examine what models learn, how their predictions differ, and where the data stop supporting inference.</p>
         <p class="atlas-byline"><a href="guide/creator.html">Rafael S. de Souza</a><span>University of Hertfordshire · UNC–Chapel Hill · UFRGS</span></p>
         <div class="atlas-actions">
           <a class="atlas-button" href="astronomy/plot_sdss_galaxy_som.html">Read the SDSS galaxy study</a>
           <a class="atlas-link" href="start/quickstart.html">Five-minute quick start</a>
         </div>
       </div>
       <figure class="atlas-hero-figure">
         <a href="astronomy/plot_sdss_galaxy_som.html">
           <img src="_static/generated/sphx_glr_plot_sdss_galaxy_som_001.svg"
                alt="Four optical colors enter a hexagonal SOM. Tile color shows the median held-out spectroscopic redshift for each supported cell." fetchpriority="high">
         </a>
         <figcaption><span>SDSS DR8 galaxies</span><span>Four optical colors · held-out spectroscopy</span></figcaption>
       </figure>
     </div>

     <section class="atlas-examples" aria-labelledby="atlas-examples-heading">
       <div class="atlas-section-head">
         <p class="atlas-kicker">From the examples</p>
         <h2 id="atlas-examples-heading">From stellar populations to photo-z calibration</h2>
       </div>
       <div class="atlas-spreads">
         <article class="atlas-example">
           <a class="atlas-example-visual" href="astronomy/plot_sdss_stellar_som.html">
             <img src="_static/generated/sphx_glr_plot_sdss_stellar_som_001.svg"
                  alt="Two hexagonal maps compare effective temperature and metallicity in the same cells of stellar color space." loading="lazy">
           </a>
           <p class="atlas-kicker">Observed stars · SDSS/SEGUE</p>
           <h3><a href="astronomy/plot_sdss_stellar_som.html">Temperature and metallicity in stellar colors</a></h3>
           <p>Read spectroscopic temperature and iron abundance on the same color map for F- and G-like dwarfs.</p>
         </article>
         <article class="atlas-example">
           <a class="atlas-example-visual" href="gallery/plot_photoz_bounds.html">
             <img src="_static/generated/sphx_glr_plot_photoz_bounds_002.svg"
                  alt="The range of compatible target fractions widens as the assumed distributional difference increases." loading="lazy">
           </a>
           <p class="atlas-kicker">Synthetic catalogue · Photo-z</p>
           <h3><a href="gallery/plot_photoz_bounds.html">How spectroscopy affects identified ranges</a></h3>
           <p>Vary the assumed difference between redshift distributions and see which target fractions remain compatible.</p>
         </article>
       </div>
     </section>

     <section class="atlas-reading" aria-labelledby="atlas-reading-heading">
       <div>
         <p class="atlas-kicker">Methods and evidence</p>
         <h2 id="atlas-reading-heading">Reproduce a figure and check its assumptions</h2>
       </div>
       <ul>
         <li><a href="examples.html">Explore astronomy and ML examples</a></li>
         <li><a href="guide/communities.html">Research communities and workflows</a></li>
         <li><a href="guide/interpretation.html">Read the scientific interpretation guide</a></li>
         <li><a href="guide/reproducibility.html">Regenerate the figures and cite the software</a></li>
         <li><a href="guide/creator.html">About</a></li>
         <li><a href="api.html">Look up a plotting function</a></li>
       </ul>
     </section>
   </div>
