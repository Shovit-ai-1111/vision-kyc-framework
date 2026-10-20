import {render,screen} from '@testing-library/react';
import {describe,it,expect} from 'vitest';
import {BarChart,LineChart} from '../components/Charts';
import {CaseResult} from '../components/Evidence';
describe('Evidence presentation',()=>{
 it('does not replace unmeasured data with a zero score',()=>{render(<BarChart title="OCR CER" data={[]}/>);expect(screen.getByText('No measured data yet.')).toBeInTheDocument();});
 it('labels development thresholds and preserves reasons',()=>{render(<CaseResult record={{result:{status:'REVIEW_REQUIRED',face:{cosine_similarity:.6,threshold:.7,calibrated:false},note:'Academic checks only.',reasons:[{code:'face_uncalibrated',message:'Collect validation pairs.',severity:'review'}]},liveness:{status:'PASSED'},consistency:{comparisons:[]}}}/>);expect(screen.getByText('Development threshold')).toBeInTheDocument();expect(screen.getByText('Collect validation pairs.')).toBeInTheDocument();});
 it('provides accessible chart labels',()=>{render(<LineChart title="Landmark metrics" series={[{label:'EAR',points:[[0,.3],[1,.1]]}]}/>);expect(screen.getByRole('img',{name:'Landmark metrics'})).toBeInTheDocument();});
});
