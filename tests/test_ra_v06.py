from pathlib import Path
import importlib.util, json, sys, tempfile, unittest, time, types, os
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT
MODEL_DIR=Path(os.environ.get('RA_ONNX_MODEL_DIR', str(ROOT/'models')))
RESULTS=ROOT/'test-results'
RESULTS.mkdir(exist_ok=True)
OFFICIAL=Path(os.environ.get('RA_OFFICIAL_REPO', str(ROOT/'upstream-reference')))
spec=importlib.util.spec_from_file_location('ra_v06',PKG/'__init__.py')
ra=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ra)

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.report=ra.RAOnnxLoadV06().load(str(MODEL_DIR/'relateanything.onnx'))
        assert cls.model['ready'], cls.report
        cls.image=np.zeros((1,64,128,3),np.float32)
        cls.image[0,:32,:,0]=1.0  # explicit RGB red, to catch channel inversion
        cls.image[0,32:,:,2]=1.0
        cls.masks=np.zeros((3,64,128),np.float32)
        cls.masks[0,2:22,4:24]=1
        cls.masks[1,36:56,70:90]=1
        cls.masks[2,0,0]=1  # rejected small instance

    def test_mask_list_and_boxes_equivalence(self):
        r,d=ra.RARegionsV06().extract([self.image],[0.5],[20],[32],masks=[self.masks[:1],self.masks[1:]])
        self.assertEqual(r['source_indices'],[0,1])
        np.testing.assert_array_equal(r['boxes'],[[4,2,24,22],[70,36,90,56]])
        self.assertEqual(json.loads(d)['filtered_small'],1)
        rb,_=ra.collect_regions(self.image,boxes_json=json.dumps(r['boxes'].tolist()))
        np.testing.assert_array_equal(rb['boxes'],r['boxes'])

    def test_bad_shapes_coordinates_ambiguous_sources(self):
        with self.assertRaisesRegex(ValueError,'source IMAGE'):
            ra.collect_regions(self.image,masks=np.zeros((2,10,10)))
        with self.assertRaisesRegex(ValueError,'exactly one'):
            ra.collect_regions(self.image,masks=self.masks,boxes_json='[]')
        with self.assertRaisesRegex(ValueError,'inside'):
            ra.collect_regions(self.image,boxes_json='[[-1,0,5,5]]')
        with self.assertRaisesRegex(ValueError,'one RGB'):
            ra.collect_regions(np.zeros((2,64,128,3)),masks=self.masks)

    def test_source_ids_filter_cap_empty(self):
        masks=np.concatenate([self.masks[2:3],self.masks[:2]])
        r,d=ra.collect_regions(self.image,masks=masks,max_regions=1)
        self.assertEqual(r['source_indices'],[1])
        self.assertEqual(json.loads(d)['dropped_by_limit'],1)
        r,_=ra.collect_regions(self.image,boxes_json='[]')
        with self.assertRaisesRegex(ValueError,'At least two'):
            ra.RAOnnxPredictV06().predict([self.model],[self.image],['above'],[20],[0.4],[1.0],regions=[r])

    def test_preprocess_official_parity(self):
        if not (OFFICIAL/'deploy/runtime.py').is_file():
            self.skipTest('Set RA_OFFICIAL_REPO to the pinned official checkout for parity verification')
        # Load the inspected official runtime without initializing its optional relsgg stack.
        deploy=types.ModuleType('deploy'); deploy.__path__=[str(OFFICIAL/'deploy')]
        sys.modules['deploy']=deploy
        offspec=importlib.util.spec_from_file_location('official_runtime',OFFICIAL/'deploy/runtime.py')
        off=importlib.util.module_from_spec(offspec); sys.modules[offspec.name]=off; offspec.loader.exec_module(off)
        ref=off.OnnxRelationHead.__new__(off.OnnxRelationHead)
        ref.img_size=448; ref.max_boxes=32; ref.vocab_mode='input'
        r,_=ra.collect_regions(self.image,masks=self.masks)
        feed=ra.make_feed(self.image[0],r['boxes'],['above','below'],self.model)
        ref._W=feed['W']; ref._alpha=feed['alpha']
        bgr=(self.image[0]*255).astype(np.uint8)[:,:,::-1]
        expected=ref.make_feed(bgr,r['boxes'])
        for key in expected:
            np.testing.assert_array_equal(feed[key],expected[key],err_msg=key)
        np.testing.assert_allclose(feed['boxes'][0,0],[14/128,12/64,20/128,20/64])
        np.testing.assert_array_equal(feed['image'][0,:,0,0],[1,0,0])

    def test_unknown_vocabulary(self):
        r,_=ra.collect_regions(self.image,masks=self.masks)
        with self.assertRaisesRegex(ValueError,'absent'):
            ra.make_feed(self.image[0],r['boxes'],['a made up relation'],self.model)

    def test_diagnostic_blocks_mismatch(self):
        changed=json.loads(json.dumps(self.model['signature']))
        changed['outputs'][0]['name']='pred_score'
        with self.assertRaisesRegex(ValueError,'Unsupported'):
            ra._check_signature(changed)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            # tiny model fixture: genuine session signature path checked separately below.
            import onnx
            from onnx import helper,TensorProto
            inp=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1])
            out=helper.make_tensor_value_info('y',TensorProto.FLOAT,[1])
            graph=helper.make_graph([helper.make_node('Identity',['x'],['y'])],'diagnostic',[inp],[out])
            m=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
            path=Path(temp)/'different.onnx'; onnx.save(m,path)
            model,report=ra.RAOnnxLoadV06().load(str(path))
            self.assertFalse(model['ready'])
            self.assertEqual(json.loads(report)['inputs'][0]['name'],'x')
            with self.assertRaisesRegex(RuntimeError,'blocked'):
                ra.RAOnnxPredictV06().predict(model,self.image,'above',20,0.4,1.0,masks=self.masks)
        # Actual model, altered metadata: session stays inspectable, inference must be blocked.
        import unittest.mock
        original_sha=ra._sha
        with unittest.mock.patch.object(ra,'_sha',side_effect=lambda p: '0'*64 if Path(p).suffix=='.json' else original_sha(p)):
            model,report=ra.RAOnnxLoadV06().load(str(MODEL_DIR/'relateanything.onnx'))
            self.assertFalse(model['ready'])
            self.assertIn('Unverified artifact',report)

    def test_decode_invalid_pairs_and_calibration(self):
        out={'pred_logits':np.array([[[0.,1.],[50,50],[50,50],[50,50],[2.,0.]]],np.float32),
             'pair_logits':np.array([[1.,0,0,0,0]],np.float32),
             'sub_idx':np.array([[0,-1,31,0,1]],np.int64),
             'obj_idx':np.array([[1,0,0,0,0]],np.int64),
             'valid_mask':np.ones((1,5),bool)}
        result=ra.decode_outputs(out,['above','below'],2,[7,9],{'a':0.5,'b':-1.},0.,20,1.)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['subject_source_idx'],7)
        self.assertAlmostEqual(result[0]['score'],0.5)
        self.assertEqual(result[0]['predicate'],'below')

    def test_real_onnx_inference(self):
        r,_=ra.collect_regions(self.image,masks=self.masks)
        start=time.perf_counter()
        result=ra.RAOnnxPredictV06().predict([self.model],[self.image],['above\nbelow'],[20],[0.0],[1.0],regions=[r])
        seconds=time.perf_counter()-start
        data=json.loads(result['result'][1])
        self.assertTrue(data['relations'])
        self.assertTrue(all(0 <= x['score'] <= 1 for x in data['relations']))
        direct=ra.RAOnnxPredictV06().predict([self.model],[self.image],['above\nbelow'],[20],[0.0],[1.0],masks=[self.masks])
        self.assertEqual(direct['result'][0],result['result'][0])
        direct_boxes=ra.RAOnnxPredictV06().predict([self.model],[self.image],['above\nbelow'],[20],[0.0],[1.0],boxes_json=[json.dumps(r['boxes'].tolist())])
        self.assertEqual(direct_boxes['result'][0],result['result'][0])
        evidence={'test':'real ONNX CPU inference on synthetic RGB and two regions',
                  'seconds_first_predict':seconds,'result':data,'relations_text':result['result'][0],
                  'limitations':'Not a SAM3 inference or full running-ComfyUI generation; no archviz quality claim.'}
        (RESULTS/'smoke-result.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    forbidden=[x for x in ['torch','transformers','relsgg'] if x in sys.modules]
    summary={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
             'forbidden_imports_present':forbidden,'passed':result.wasSuccessful() and not forbidden}
    (RESULTS/'test-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary))
    sys.exit(0 if summary['passed'] else 1)
