Deep Learning for Computer Vision (CSE3292) January 2026 

Project Overview 

Few Indian computer vision datasets currently exist in domains such as traffic surveillance, action recognition, and multilingual text recognition. While these datasets provide valuable initial resources for research and development, they remain limited in scale, coverage, and contextual diversity. Most available datasets capture only a narrow subset of Indian environments, often focusing on specific cities, controlled settings, or restricted visual scenarios. As a result, they fail to adequately represent the vast geographic, cultural, demographic, and infrastructural diversity that characterizes India. The limited scope of these datasets restricts their effectiveness for building robust, fair, and context-aware AI systems. This highlights a critical need for large-scale, comprehensive, and ethically curated Indian datasets. The objective of the IndiVision project is to create a large-scale, high-quality computer vision dataset that accurately represents Indian environments, demographics, infrastructure, and cultural contexts.

1 Motivation 

Most computer vision research relies on Western-centric datasets (ImageNet, COCO, CelebA). However, these datasets often fail to capture: 

    Indian architectural diversity (temples, forts, regional styles) 

    Cultural artifacts (textiles, jewelry, handicrafts) 

    Agricultural contexts (crop varieties, farming practices) 

    Urban/rural visual environments specific to India 

    Regional language scripts in natural scenes 

    Indian wildlife and biodiversity 

Few key differences between Western and Indian datasets and their impact on computer vision applications are highlighted in Table 1. Models trained on Western datasets may: 

    Under-perform in Indian real-world deployment 

    Fail in crowded or chaotic scenes 

    Exhibit fairness bias across demographics 

    Miss culturally specific objects or actions 

By creating Indian-context datasets, you will contribute to more inclusive computer vision research while learning data collection best practices. Example domains where you can contribute are traffic, healthcare, agriculture, rural monitoring, surveillance, multilingual text, and scene recognition. These datasets, representing diverse Indian urban and rural environments, can enable robust object detection and fairness-aware learning.

Table 1: Key Differences between Western-centric and Indian-context Vision Datasets 
Theme	Western-centric	Indian-context	Impact

Demographics & Appearance
	

Mostly lighter skin tones; Western clothing
	

Wider skin tone diversity; Traditional clothing
	

Face recognition bias; Pose estimation errors

Environment & Infrastructure
	

Well-marked roads & lanes; Organized sidewalks; Structured urban scenes; Standardized signage
	

Crowded streets; Informal infrastructure; Mixed pedestrian-vehicle traffic; Irregular signage
	

Autonomous driving failures; Object detection confusion; Scene understanding errors

Transportation Patterns & Vehicles
	

Mostly cars, buses, bicycles; Limited animal presence
	

Two/three-wheelers; Animals on roads
	

Poor vehicle detection; Traffic prediction inaccuracies

Architecture & Urban Visual Patterns
	

Uniform building styles; Fewer informal settlements; Cleaner backgrounds
	

Dense, irregular architecture; Slums, street vendors; Visually complex backgrounds
	

Scene segmentation struggles; Object localization errors

Language, Scripts & Text Recognition
	

Mostly English text; Clean fonts and signage
	

Multiple scripts; Mixed, hand-painted multilingual signs
	

OCR failures; Poor text detection

Object & Scene Categories
	

Supermarkets; Baseball bats, skateboards, Snow gear, fireplaces, barbeques
	

Local shops; School uniforms, tiffins, farming tools
	

Poor object coverage; Label mismatch; Class imbalance

Weather & Visual Conditions
	

Seasonal snow; Cleaner air
	

Harsh sunlight, monsoon rain; Haze, dust, pollution
	

Domain shift; Lower detection accuracy

Annotation Style & Label Semantics
	

Highly standardized labels; Clean taxonomy
	

Need culturally sensitive labels; Localized category names
	

Cross-dataset incompatibility; Training noise

2 Dataset Requirements 

    Group Formation: 5 students per group 

    Dataset Specifications: 

        Size: Minimum 1000 images, target 2000-2500 images OR minimum 250 videos, target 500-700 

        Quality: Consistent resolution, proper labeling, documented sources 

        Accessibility: Use publicly available sources (YouTube videos, Flickr Creative Commons, government portals, your own photos) 

        Feasibility: Avoid medical/sensitive data requiring institutional approval 

        Inspiration: Model after established dataset structures (see examples below) 

Why This Size is Sufficient: Since your Phase 2 work focuses on diagnostic analysis using pretrained models (not training from scratch), 2000-2500 images provide enough data to reveal meaningful patterns in model behavior. You will be investigating how models work, not achieving state-of-the-art accuracy.

3 Example Dataset Ideas 

You can support the computer vision community by creating high-quality Indian datasets that capture the country's diverse environments, cultures, and real-world conditions. Such datasets improve model fairness, robustness, and relevance, enable India-specific AI applications, and help bridge gaps caused by Western-centric data, advancing practical vision research. Some examples are outlined below for your understanding.

    Indian Architectural Styles Dataset 

        Inspired by: Places365 

        Classes: Mughal (Taj Mahal style), Dravidian (South Indian temples), Colonial (British-era), Modern Indo-Saracenic, Regional forts 

        Sources: Google Images (Creative Commons), Wikimedia, personal photography 

    Regional Script Detection 

        Inspired by: COCO-Text, Street View Text 

        Classes: Devanagari, Tamil, Telugu, Bengali, Gujarati scripts in natural scenes (shop signs, posters) 

        Sources: YouTube street walk videos, Google Street View 

    Indian Agricultural Crop Classification 

        Inspired by: Plant Village 

        Classes: Rice, wheat, cotton, sugarcane, pulses at different growth stages 

        Sources: Agricultural university databases, YouTube farming channels, Krishi Vigyan Kendras 

    Indian Traffic Scene Understanding 

        Inspired by: Cityscapes 

        Classes: Segmentation of vehicles (auto-rickshaws, two-wheelers), pedestrians, cattle, street vendors 

        Sources: Dashcam footage, traffic camera feeds (if public) 

    Traditional Textile Pattern Recognition 

        Inspired by: Describable Textures Dataset (DTD) 

        Classes: Banarasi silk, Kanjeevaram, Ikat, Bandhani, Block prints 

        Sources: E-commerce sites, museum archives, textile exhibitions 

    Indian Wildlife in Natural Habitats 

        Inspired by: iNaturalist 

        Classes: Bengal tiger, Indian elephant, peacock, regional birds, leopards 

        Sources: Wildlife photography databases, sanctuary camera traps (public datasets) 

    Street Food Classification 

        Inspired by: Food-101 

        Classes: Pani puri, dosa, samosa, jalebi, regional specialties 

        Sources: YouTube food videos, Instagram (with permission), personal photos 

    Indian Festival Scenes 

        Inspired by: Event Net 

        Classes: Diwali decorations, Holi celebrations, Durga Puja pandals, regional harvest festivals 

        Sources: Public YouTube videos, Flickr Creative Commons 

Other popular datasets which can be used as a reference to create similar Indian datasets are KITTI, Waymo, UCF101, SVHN, VQA etc. These datasets can be used as baselines for comparison, examples of what biases to avoid and inspiration for annotation standards.

4 Annotation Guidelines 

Dataset quality depends on consistent and thoughtful annotation. Follow these practices: 

4.1 Create Clear Class Definitions: 

    Write 2-3 sentence descriptions for each category 

    Include 3-5 visual examples showing typical instances 

    Document edge cases (e.g., "What if an image contains multiple classes?") 

4.2 Assign Explicit Labels: 

    Labels must reflect what is visibly present, not assumptions 

    Do not guess hidden or occluded objects, annotate if > 30-40% is visible 

    Annotate all eligible objects or actions, not only prominent ones 

    Semantic segmentation datasets: masks should follow object boundaries accurately 

    Activity recognition datasets: mark start and end frames of each action, assign multiple labels if actions overlap 

    Similarly, for task-specific datasets, define clear annotation guidelines 

4.3 Multi-Annotator Protocol (Recommended): 

    Have 2 team members independently label the same 100 images 

    Compare labels and discuss disagreements 

    Refine class definitions based on what you learn 

    Once agreement is high, split remaining annotation work 

4.4 Maintain Annotation Metadata: 

Create a simple CSV file tracking each image: image_id, filename, label, annotator, date, source 1, img_001.jpg, Banarasi, Alice, 2026-01-25, wikimedia 2, img_002.jpg, Kanjeevaram, Bob, 2026-01-26, flickr This documentation makes your dataset reproducible and helps identify patterns later.

5 Tools You Can Use 

    Simple classification: Google Sheets, Label Studio 

    Bounding boxes/segmentation: CVAT, Roboflow 

6 Ethical Guidelines 

    Consent: Only use publicly available data or your own photographs 

    Privacy: Blur faces in crowd scenes unless explicit consent obtained 

    Licensing: Document image sources and respect Creative Commons licenses 

    Bias: Ensure geographic and demographic diversity within your dataset 

    Attribution: Maintain metadata citing all sources 

7 Deliverables 

7.1 Dataset Pre-Curation Presentation 

    Format: 3-4 slides, 5-minute presentation per group 

    Required Content: 

        Slide 1 Motivation: What existing dataset inspired you? Why is Indian context important for this domain? 

        Slide 2 Dataset Design: 

            Sources identified 

            Classes/categories with brief definitions 

            Sample images (10-15 examples) 

            Expected dataset size and class distribution 

            Naming convention 

        Slide 3 Annotation Strategy: 

            How did you define class boundaries (are they clear or ambiguous) 

            Annotation protocol (single annotator vs multi-annotator) 

            How will edge cases be handled 

            Tools you will use 

        Slide 4 Feasibility and Ethics: 

            Ethical considerations (consent, privacy, licensing) 

            How your team can collect this data in 6 weeks 

7.2 Gold-standard Dataset 

Submit your curated gold-standard dataset in a clean, reproducible folder structure as: 

    Dataset Name/: 

            images (.jpg/.png)/ or videos (mp4/avi/mov)/ 

            annotations/ 

            metadata/ 

            splits/(train/val/test) 

            docs/ 

            README.md 

            LICENSE 

8 Grading (10 marks) 

    Domain selection and problem relevance (2 marks) 

    Data diversity and ethical considerations (2 marks) 

    Dataset design quality and annotation (6 marks) 

9 Timelines 

    Dataset pre-curation presentation: February 9, 2026 

    Dataset submission: March 16, 2026