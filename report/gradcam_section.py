# Executed by build_report.py (shares its helpers: H2, P, FIG, BUL, ASSETS).
# Written after looking at every image; the observations below describe what the maps show.
GC = ASSETS / "figures"

H2("12.1 ResNet-50 fine-tuned (best model)")
FIG(GC / "gradcam_resnet50_finetune" / "gradcam_resnet50_finetune_confident_correct.png",
    "ResNet-50 fine-tuned, confident correct predictions (probability 0.99).", 5.6)
FIG(GC / "gradcam_resnet50_finetune" / "gradcam_resnet50_finetune_barely_correct.png",
    "ResNet-50 fine-tuned, barely correct predictions (probability 0.06-0.07).", 5.6)
FIG(GC / "gradcam_resnet50_finetune" / "gradcam_resnet50_finetune_confident_wrong.png",
    "ResNet-50 fine-tuned, confident wrong predictions (probability 0.95-0.97).", 5.6)
FIG(GC / "gradcam_resnet50_finetune" / "gradcam_resnet50_finetune_confused_pair.png",
    "ResNet-50 fine-tuned, the most frequent mistake: Great Grey Shrike predicted as Loggerhead Shrike.", 5.6)
P("**Confident correct.** The heat sits on the bird's body in all four images: the streaked chest and "
  "belly of the Harris's Sparrow, the body and head of the Black Tern, the yellow breast of the Scott "
  "Oriole, and the back and head of the Green-tailed Towhee. Branches, sky and gravel are ignored. "
  "The model is using plumage colour and pattern on the body, which is also how people tell these "
  "species apart, so the high confidence is backed by sensible evidence.")
P("**Barely correct.** These are right, but with a probability of only 6-7%. In three of the four images "
  "the hottest region is **not on the bird**: foliage in the top corner (Bohemian Waxwing), an empty "
  "lower-left corner (Herring Gull) and leaves at the top left (Fox Sparrow). Only the Fish Crow map "
  "falls on the bird (its head). These are the hardest correct test images (low contrast, unusual "
  "crops or poses), and the maps suggest that when the bird itself gives weak evidence the model "
  "falls back on context. Such predictions are fragile: they are right for weak or partly "
  "irrelevant reasons.")
P("**Confident wrong.** In most cases the model is looking at the bird, so the mistake is not caused by "
  "the background: for the Hooded Merganser predicted as Red-breasted Merganser the heat is on the head "
  "and chest, and for the Worm-eating Warbler predicted as Ovenbird it is on the streaked breast "
  "(the bird is cut off at the image edge, and both species have streaked breasts). For the "
  "Rusty Blackbird predicted as Brewer's Blackbird, part of the heat falls on an empty area beside "
  "the body, and both are uniformly dark birds. The Green Kingfisher predicted as Ringed "
  "Kingfisher has heat on the chest; the photo is ambiguous to us, so an atypical or mislabelled "
  "image cannot be ruled out. The errors are therefore between near-identical plumages, which agrees "
  "with the confusion pairs in Section 11. The 0.95-0.97 confidence also shows the model is "
  "overconfident when it is wrong.")
P("**Confused pair.** For the Great Grey Shrike / Loggerhead Shrike mistake the confidence is lower "
  "(0.48-0.70) and the heat covers the body and wing patch, in one image the head and wing. The two "
  "species differ mainly in the width of the black face mask and the bill, which are only a few "
  "pixels in these photos (in two of them the bird fills a small part of the frame). Our reading "
  "is that at 224 px the model cannot resolve that cue and relies on the shared grey-and-black "
  "body pattern; higher-resolution inputs would test this.")

H2("12.2 ScratchCNN (baseline trained from scratch)")
FIG(GC / "gradcam_scratch_cnn" / "gradcam_scratch_cnn_confident_correct.png",
    "ScratchCNN, confident correct predictions (probability 0.83-0.94).", 5.6)
FIG(GC / "gradcam_scratch_cnn" / "gradcam_scratch_cnn_barely_correct.png",
    "ScratchCNN, barely correct predictions (probability 0.02-0.03).", 5.6)
FIG(GC / "gradcam_scratch_cnn" / "gradcam_scratch_cnn_confident_wrong.png",
    "ScratchCNN, confident wrong predictions (probability 0.71-0.93).", 5.6)
FIG(GC / "gradcam_scratch_cnn" / "gradcam_scratch_cnn_confused_pair.png",
    "ScratchCNN, the most frequent mistake: Frigatebird predicted as Pomarine Jaeger.", 5.6)
P("**Confident correct.** The maps fall on high-contrast colour patches: the red head and black-and-white "
  "back of the Red-bellied Woodpecker and the striped flank of the Hooded Merganser. In the first "
  "merganser image part of the heat is on the **water reflection**, which shows the network is "
  "responding to colour and texture blobs rather than to the bird as a whole.")
P("**Barely correct.** With probabilities of 2-3%, the maps point at branches, ground and leaves in three "
  "of four images (the lower-right branch tip for the Spotted Catbird, the ground for the White-crowned "
  "Sparrow, leaves for the Warbling Vireo). These are essentially lucky guesses.")
P("**Confident wrong.** Here the failure mode is clear: for the Horned Puffin and the Eared Grebe, both "
  "predicted as Hooded Merganser, the heat is on the **rippling water**, not on the bird. The network "
  "has learned the shortcut \"water scene -> merganser\". The other two errors follow colour alone: "
  "the Scott Oriole's yellow-and-black body is read as Evening Grosbeak, and the Green Violetear's "
  "blue-green plumage as Indigo Bunting.")
P("**Confused pair.** Frigatebirds are predicted as Pomarine Jaegers with low confidence (0.16-0.20). The "
  "heat lies on the dark wings against the sky; the forked tail and white throat patch that "
  "identify a frigatebird are not used. The faint rectangular frame visible in several maps is a border "
  "effect of zero padding in this small network, not image content.")

H2("12.3 What the two models tell us")
BUL(["**ResNet-50 looks at the bird, ScratchCNN at colour and scenery.** The pretrained model concentrates on "
     "the body and plumage and ignores the background when it is confident, and its errors are between "
     "near-identical species. The scratch model uses colour blobs and water or ground texture, so its "
     "errors are between unrelated species (puffin and grebe called merganser). This is consistent "
     "with the scores (80.7% vs 17.3%) and with the idea that ImageNet pretraining provides object-level "
     "features that 4,800 images cannot teach.",
     "**Low-confidence correct predictions are the weak spot of both models.** In both, the maps for the "
     "barely correct group often leave the bird, so accuracy overstates how well the evidence supports "
     "some predictions.",
     "**Limits of this analysis.** Grad-CAM shows where evidence comes from, not why a feature matters. "
     "Each group is four images selected automatically by confidence, so these are illustrations, not "
     "statistics, and our reading of species identity in the photos is that of non-experts."])
