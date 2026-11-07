:html_theme.sidebar_secondary.remove: true

ModelAtlas
==========

.. toctree::
   :maxdepth: 2
   :hidden:

   Start here <start/index>
   Example gallery <gallery/index>
   User guide <guide/index>
   API reference <api>
   Development <development/index>

.. raw:: html

   <div class="atlas-home">
     <div class="atlas-hero">
       <div class="atlas-hero-copy">
         <p class="atlas-kicker">Scientific Python <span aria-hidden="true">/</span> Research preview</p>
         <h1>Scientific figures<br> from fitted models</h1>
         <p class="atlas-lede">Trace fitted prototypes, compare aligned neural networks, and show how spectroscopic selection affects what a redshift analysis can establish.</p>
         <div class="atlas-actions">
           <a class="atlas-button" href="start/quickstart.html">Five-minute quick start</a>
           <a class="atlas-link" href="gallery/index.html">Browse the gallery <span aria-hidden="true">↗</span></a>
         </div>
       </div>
       <figure class="atlas-hero-figure">
         <a href="gallery/plot_hexagonal_som.html">
           <img src="_static/generated/sphx_glr_plot_hexagonal_som_001.svg" width="720" height="450"
                alt="Five annual-activity inputs connect to an oblique 8 by 8 hexagonal SOM. Deep blue to yellow encodes central annual sunspot counts." fetchpriority="high">
         </a>
         <figcaption><span>01 · Historical solar activity</span><span>Hexagonal SOM · NOAA annual series</span></figcaption>
       </figure>
     </div>

     <section class="atlas-examples" aria-labelledby="atlas-examples-heading">
       <div class="atlas-section-head">
         <p class="atlas-kicker">From the gallery</p>
         <h2 id="atlas-examples-heading">What each figure shows</h2>
       </div>
       <div class="atlas-spreads">
         <article class="atlas-example">
           <a class="atlas-example-visual" href="gallery/plot_prototype_graph.html">
             <img src="_static/generated/sphx_glr_plot_prototype_graph_001.svg" width="680" height="520"
                  alt="An irregular graph of learned Wine prototypes, with larger tiles for greater occupancy." loading="lazy">
           </a>
           <p class="atlas-kicker">02 · Prototype graphs</p>
           <h3><a href="gallery/plot_prototype_graph.html">Learned neighbors and projected distances</a></h3>
           <p>See which prototypes the fitted graph connects and measure how much its layout distorts distances.</p>
         </article>
         <article class="atlas-example">
           <a class="atlas-example-visual" href="gallery/plot_local_explanations.html">
             <img src="_static/generated/sphx_glr_plot_local_explanations_001.svg" width="720" height="660"
                  alt="Two oblique hidden layers with colored paths showing contributions to a Wine classifier logit." loading="lazy">
           </a>
           <p class="atlas-kicker">03 · Neural explanations</p>
           <h3><a href="gallery/plot_local_explanations.html">Contributions to one prediction</a></h3>
           <p>Follow changes in a fitted logit from a specified baseline, with fixed neuron positions across observations.</p>
         </article>
       </div>
     </section>

     <section class="atlas-reading" aria-labelledby="atlas-reading-heading">
       <div>
         <p class="atlas-kicker">Methods and evidence</p>
         <h2 id="atlas-reading-heading">Reproduce a figure and check its assumptions</h2>
       </div>
       <ul>
         <li><a href="gallery/plot_aligned_networks.html">Align independent neural networks</a></li>
         <li><a href="gallery/plot_photoz_bounds.html">Vary the assumptions in a synthetic photo-z analysis</a></li>
         <li><a href="guide/interpretation.html">Read the scientific interpretation guide</a></li>
         <li><a href="guide/reproducibility.html">Regenerate the figures and cite the software</a></li>
       </ul>
     </section>
   </div>
